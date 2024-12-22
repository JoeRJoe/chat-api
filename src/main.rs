use std::fs::File;
use std::path::Path;
use std::sync::mpsc;

use file_chunker::FileChunker;
use kalosm::language::{Bert, Chat, EmbedderExt, Llama, LlamaSource, TextStream};
use notify::{Event, RecursiveMode, Result, Watcher};
use pgvector::Vector;
use rocket::fairing::{Fairing, Info, Kind};
use rocket::serde::Serialize;
use rocket::{futures::lock::Mutex, serde::json::Json, State};
use rocket::{Orbit, Rocket};
use rocket_db_pools::{deadpool_postgres, Database};

#[macro_use]
extern crate rocket;

#[launch]
async fn rocket() -> _ {
    rocket::build()
        .manage(ChatWrapper {
            chat: Mutex::new(
                Chat::builder(
                    Llama::builder()
                        .with_source(LlamaSource::llama_3_2_3b_chat())
                        .build()
                        .await
                        .unwrap(),
                )
                .build(),
            ),
        })
        .attach(PgVector::init())
        .attach(Listener)
        .mount("/", routes![generate_text])
}

struct Listener;

impl Fairing for Listener {
    fn info(&self) -> Info {
        Info {
            name: "Listener",
            kind: Kind::Liftoff,
        }
    }

    fn on_liftoff<'life0, 'life1, 'async_trait>(
        &'life0 self,
        _rocket: &'life1 Rocket<Orbit>,
    ) -> ::core::pin::Pin<
        Box<dyn ::core::future::Future<Output = ()> + ::core::marker::Send + 'async_trait>,
    >
    where
        'life0: 'async_trait,
        'life1: 'async_trait,
        Self: 'async_trait,
    {
        Box::pin(async move {
            let (sender, receiver) = mpsc::channel::<Result<Event>>();
            let mut watcher = notify::recommended_watcher(sender).unwrap();
            watcher
                .watch(Path::new("./documents"), RecursiveMode::Recursive)
                .unwrap();
            let db = PgVector::fetch(_rocket).unwrap();
            let vector = db.get().await.unwrap();
            let embedder = Bert::new().await.unwrap();

            info!("Listener Initialized");
            for event in receiver {
                match event {
                    Ok(event) => {
                        if let notify::EventKind::Create(_) = event.kind {
                            for path in event.paths {
                                let file = File::open(&path).unwrap();
                                let chunker = FileChunker::new(&file).unwrap();
                                let bytes_vector = chunker.chunks(1024, None).unwrap();

                                for bytes in bytes_vector {
                                    let text = pdf_extract::extract_text_from_mem(bytes)
                                        .expect("Failed to extract text");

                                    let embeddings = embedder
                                        .embed(text.as_str())
                                        .await
                                        .expect("Failed to embed text");

                                    vector.execute("INSERT INTO document (embedding, text, name) VALUES ($1, $2, $3)", 
                                &[
                                    &Vector::from(embeddings.to_vec()),
                                    &text,
                                    &path.to_str().unwrap().split("/").last().expect("Failed to get document name")
                                ])
                                .await
                                .expect("Failed to insert document");
                                }
                            }
                        }
                    }
                    Err(e) => {
                        error!("Error: {:?}", e);
                    }
                }
            }
        })
    }
}

#[derive(Database)]
#[database("pgvector")]
struct PgVector(deadpool_postgres::Pool);

struct ChatWrapper {
    chat: Mutex<Chat>,
}

#[derive(Serialize)]
#[serde(crate = "rocket::serde")]
struct GeneratedText<'a> {
    prompt: &'a str,
    text: String,
}

#[get("/<prompt>")]
async fn generate_text<'a>(
    chat_wrapper: &State<ChatWrapper>,
    prompt: &'a str,
) -> Json<GeneratedText<'a>> {
    Json(GeneratedText {
        prompt,
        text: chat_wrapper
            .chat
            .lock()
            .await
            .add_message(prompt)
            .all_text()
            .await,
    })
}
