use std::path::Path;
use std::sync::mpsc;
use std::vec;

use dotenv::dotenv;
use indicatif::ProgressBar;
use kalosm::language::{Bert, EmbedderExt};
use notify::event::{ModifyKind, RenameMode};
use notify::{Event, RecursiveMode, Result, Watcher};
use pgvector::Vector;
use rocket::fairing::{self, Fairing, Info, Kind};
use rocket::Rocket;
use rocket_db_pools::deadpool_postgres::Object;
use rocket_db_pools::{deadpool_postgres, Database};

#[macro_use]
extern crate rocket;

#[launch]
async fn rocket() -> _ {
    rocket::build()
        .attach(PgVector::init())
        .attach(Listener)
        .mount("/", routes![])
}

struct Listener;

impl Fairing for Listener {
    fn info(&self) -> Info {
        Info {
            name: "Listener",
            kind: Kind::Ignite,
        }
    }

    fn on_ignite<'life0, 'async_trait>(
        &'life0 self,
        rocket: Rocket<rocket::Build>,
    ) -> ::core::pin::Pin<
        Box<
            dyn ::core::future::Future<Output = rocket::fairing::Result>
                + ::core::marker::Send
                + 'async_trait,
        >,
    >
    where
        'life0: 'async_trait,
        Self: 'async_trait,
    {
        Box::pin(async move {
            dotenv().ok();
            let path = std::env::var("EMBEDDINGS_PATH").expect("EMBEDDINGS_PATH not set in ENV");
            let (sender, receiver) = mpsc::channel::<Result<Event>>();
            let mut watcher = notify::recommended_watcher(sender).unwrap();
            watcher
                .watch(Path::new(path.as_str()), RecursiveMode::Recursive)
                .unwrap();
            let db = PgVector::fetch(&rocket).unwrap();
            let vector = db.get().await.unwrap();
            let embedder = Bert::new().await.unwrap();

            initialize_db(&vector).await;

            info!("Listener Initialized");
            for event in receiver.into_iter().flatten() {
                match event.kind {
                    notify::EventKind::Create(_)
                    | notify::EventKind::Modify(ModifyKind::Name(RenameMode::To)) => {
                        for path in event.paths {
                            if path.extension().unwrap() != "pdf" {
                                info!("{} is not a pdf file", path.display());
                                continue;
                            }

                            info!("Embedding {}...", path.display());
                            let bytes = std::fs::read(&path).expect("Failed to read file");
                            let text = pdf_extract::extract_text_from_mem(&bytes)
                                .expect("Filed to extract text");
                            let words = text.split_whitespace().collect::<Vec<_>>();
                            let mut chunks: Vec<String> = Vec::new();
                            for chunk in words.chunks(100) {
                                    chunks.push(chunk.join(" "));
                            }

                            
                            let main_bar = ProgressBar::new(chunks.len() as u64);
                            main_bar.inc(0);
                            
                            for chunk in chunks {
                                if chunk.is_empty() {
                                    continue;
                                }
                                
                                let embeddings = match embedder.embed(&chunk).await {
                                    Ok(embeddings) => embeddings,
                                    Err(_) => {
                                        info!("Failed to embed a chunk");
                                        continue;
                                    }
                                };

                                let document_name = path.to_str().unwrap().split("/").last().expect("Failed to get document name");
                                
                                vector.execute("INSERT INTO document (embedding, text, name, embedded_name) VALUES ($1, $2, $3, $4)", 
                                &[
                                    &Vector::from(embeddings.to_vec()),
                                    &chunk,
                                    &document_name,
                                    &Vector::from(embedder.embed(document_name).await.expect("Failed to embed document name").to_vec())
                                    ])
                                    .await
                                    .expect("Failed to insert document");

                                main_bar.inc(1);

                            }
                            main_bar.finish();
                            info!("Embedded {}", path.display());
                        }
                    }
                    notify::EventKind::Remove(_)
                    | notify::EventKind::Modify(ModifyKind::Name(RenameMode::From)) => {
                        for path in event.paths {
                            if path.extension().unwrap() != "pdf" {
                                info!("{} is not a pdf file", path.display());
                                continue;
                            }

                            let name = path
                                .to_str()
                                .unwrap()
                                .split("/")
                                .last()
                                .expect("Failed to get document name");

                            vector
                                .execute("DELETE FROM document WHERE name = $1", &[&name])
                                .await
                                .expect("Failed to delete document");
                            info!("Removed {}", path.display());
                        }
                    }
                    _ => {}
                }
            }
            fairing::Result::Ok(rocket)
        })
    }
}

async fn initialize_db(vector: &Object) {
    vector
        .execute("CREATE EXTENSION IF NOT EXISTS vector", &[])
        .await
        .expect("Failed to create extension vector");
    vector
        .execute(
            "CREATE TABLE IF NOT EXISTS document (
                id SERIAL PRIMARY KEY,
                embedding vector(384),
                text TEXT,
                name TEXT,
                embedded_name vector(384)
            )",
            &[],
        )
        .await
        .expect("Failed to create table");
}

#[derive(Database)]
#[database("pgvector")]
struct PgVector(deadpool_postgres::Pool);
