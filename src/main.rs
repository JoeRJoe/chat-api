use std::path::Path;
use std::sync::mpsc;

use kalosm::language::{Bert, EmbedderExt};
use notify::{Event, RecursiveMode, Result, Watcher};
use pgvector::Vector;
use rocket::fairing::{self, Fairing, Info, Kind};
use rocket::Rocket;
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
            let (sender, receiver) = mpsc::channel::<Result<Event>>();
            let mut watcher = notify::recommended_watcher(sender).unwrap();
            watcher
                .watch(Path::new("./documents"), RecursiveMode::Recursive)
                .unwrap();
            let db = PgVector::fetch(&rocket).unwrap();
            let vector = db.get().await.unwrap();
            let embedder = Bert::new().await.unwrap();

            info!("Listener Initialized");
            for event in receiver {
                match event {
                    Ok(event) => {
                        if let notify::EventKind::Create(_) = event.kind {
                            for path in event.paths {
                                let bytes = std::fs::read(&path).expect("Failed to read file");
                                let text = pdf_extract::extract_text_from_mem(&bytes)
                                    .expect("Failed to extract text");
                                let text_chunked: Vec<_> = text.split("\n").collect();

                                for t in text_chunked {
                                    if !t.is_empty() {
                                        let embeddings =
                                            embedder.embed(t).await.expect("Failed to embed text");

                                        vector.execute("INSERT INTO document (embedding, text, name) VALUES ($1, $2, $3)", 
                                        &[
                                            &Vector::from(embeddings.to_vec()),
                                            &t,
                                            &path.to_str().unwrap().split("/").last().expect("Failed to get document name")
                                        ])
                                        .await
                                        .expect("Failed to insert document");
                                    }
                                }
                            }
                        }
                    }
                    Err(e) => {
                        error!("Error: {:?}", e);
                    }
                }
            }
            fairing::Result::Ok(rocket)
        })
    }
}

#[derive(Database)]
#[database("pgvector")]
struct PgVector(deadpool_postgres::Pool);
