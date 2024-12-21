use std::path::Path;
use std::sync::mpsc;

use kalosm::language::{Bert, Chat, EmbedderExt, Llama, LlamaSource, TextStream};
use notify::{Event, RecursiveMode, Result, Watcher};
use pgvector::{self, Vector};
use rocket::fairing::AdHoc;
use rocket::serde::Serialize;
use rocket::{futures::lock::Mutex, serde::json::Json, State};
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
        .attach(AdHoc::on_liftoff("Listener", |_| {
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
        }))
        .mount("/", routes![generate_text])
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
