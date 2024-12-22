use std::vec;

use kalosm::language::{Bert, Chat, EmbedderExt, Llama, LlamaSource, TextStream};
use pgvector::Vector;
use rocket::serde::Serialize;
use rocket::{futures::lock::Mutex, serde::json::Json, State};
use rocket_db_pools::{deadpool_postgres, Connection, Database};

#[macro_use]
extern crate rocket;

#[launch]
async fn rocket() -> _ {
    rocket::build()
        .manage(EmbedderWrapper {
            embedder: Bert::new().await.unwrap(),
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
        .mount("/", routes![generate_text])
}

#[derive(Database)]
#[database("pgvector")]
struct PgVector(deadpool_postgres::Pool);

struct ChatWrapper {
    chat: Mutex<Chat>,
}

struct EmbedderWrapper {
    embedder: Bert,
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
    embedder_wrapper: &State<EmbedderWrapper>,
    chat_wrapper: &State<ChatWrapper>,
    prompt: &'a str,
) -> Json<GeneratedText<'a>> {
    let embedded_prompt_vector = Vector::from(
        embedder_wrapper
            .embedder
            .embed(prompt)
            .await
            .expect("Failed to embed prompt")
            .to_vec(),
    );
    let context = format!(
        "Rispondi in maniera semplice e concisa. Contesto : {}. Domanda: {}",
        vector
            .query(
                "SELECT text FROM document ORDER BY embedding <-> $1 LIMIT 5",
                &[&embedded_prompt_vector],
            )
            .await
            .expect("Failed to find nearest neighbors")
            .iter()
            .map(|row| row.get::<_, String>(0) + "\n")
            .collect::<Vec<_>>()
            .join(" "),
        prompt
    );
    Json(GeneratedText {
        prompt,
        text: chat_wrapper
            .chat
            .lock()
            .await
            .add_message(context)
            .all_text()
            .await,
    })
}
