
use kalosm::language::{Chat, Llama, LlamaSource, TextStream};
use rocket::serde::Serialize;
use rocket::{futures::lock::Mutex, serde::json::Json, State};
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
        .mount("/", routes![generate_text])
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
