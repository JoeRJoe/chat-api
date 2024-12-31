use std::vec;

use kalosm::language::{Bert, Chat, EmbedderExt, Llama, TextStream};
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
                    Llama::new_chat()
                        .await
                        .unwrap(),
                )
                .with_system_prompt(
                    "Questo assistente risponde in maniera precisa, senza inventare nulla, in italiano,
                    prendendo principalmente le informazioni contenute nel contesto"
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

#[post("/", data = "<prompt>")]
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
        "Contesto : {}. Domanda: {}",
        vector
            .query(
                "SELECT text FROM document WHERE embedded_name <-> $1 > 0.5 ORDER BY embedding <=> $1 LIMIT 3",
                &[&embedded_prompt_vector],
            )
            .await
            .expect("Failed to find nearest neighbors")
            .iter()
            .map(|row| row.get::<_, String>(0))
            .collect::<Vec<_>>()
            .join("\n"),
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
