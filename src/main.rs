use std::path::Path;
use std::sync::mpsc;

use kalosm::language::{Bert, Chat, EmbedderExt, Llama, LlamaSource, TextStream};
use notify::{Event, RecursiveMode, Result, Watcher};
use pgvector::{self, Vector};
use rocket::fairing::{Fairing, Info, Kind};
use rocket::serde::Serialize;
use rocket::{futures::lock::Mutex, serde::json::Json, State};
use rocket::{Orbit, Rocket};
use rocket_db_pools::{deadpool_postgres, Connection, Database};

#[macro_use]
extern crate rocket;

#[launch]
async fn rocket() -> _ {
    rocket::build()
        .manage(EmbeddingWrapper {
            model: Mutex::new(Bert::new().await.unwrap()),
        })
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

#[rocket::async_trait]
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
            info!("Listener Initialized");

            for event in receiver {
                match event {
                    Ok(event) => {
                        if let notify::EventKind::Create(_) = event.kind {
                            info!("New file created");
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

struct EmbeddingWrapper {
    model: Mutex<Bert>,
}

#[derive(Serialize)]
#[serde(crate = "rocket::serde")]
struct GeneratedText<'a> {
    prompt: &'a str,
    text: String,
}

#[get("/<prompt>")]
async fn generate_text<'a>(
    vector: Connection<PgVector>,
    chat_wrapper: &State<ChatWrapper>,
    embedding_wrapper: &State<EmbeddingWrapper>,
    prompt: &'a str,
) -> Json<GeneratedText<'a>> {
    let embedding = embedding_wrapper
        .model
        .lock()
        .await
        .embed(prompt)
        .await
        .unwrap();

    vector
        .execute(
            "INSERT INTO items (embedding, testo) VALUES ($1, $2)",
            &[&Vector::from(embedding.to_vec()), &prompt],
        )
        .await
        .unwrap();

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
