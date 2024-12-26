use std::path::Path;
use std::sync::mpsc;

use indicatif::ProgressBar;
use kalosm::language::{Bert, EmbedderExt};
use notify::event::{ModifyKind, RenameMode};
use notify::{Event, RecursiveMode, Result, Watcher};
use pgvector::Vector;
use rocket::fairing::{self, Fairing, Info, Kind};
use rocket::Rocket;
use rocket_db_pools::{deadpool_postgres, Database};
use dotenv::dotenv;

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
                .watch(
                    Path::new(path.as_str()),
                    RecursiveMode::Recursive,
                )
                .unwrap();
            let db = PgVector::fetch(&rocket).unwrap();
            let vector = db.get().await.unwrap();
            let embedder = Bert::new().await.unwrap();

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
                            let chunks = text.split("\n\n").collect::<Vec<_>>();
                            let bar = ProgressBar::new(chunks.len() as u64);
                            let mut neighbors = vec!["".to_string(); chunks.len()];

                            for (i, chunk_a) in chunks.iter().enumerate() {
                                let embed_a =
                                    embedder.embed(chunk_a).await.expect("Failed to embed text");
                                for chunk_b in chunks.iter().skip(i + 1) {
                                    let embed_b = embedder
                                        .embed(chunk_b)
                                        .await
                                        .expect("Failed to embed text");
                                    let is_neighbor = embed_a.cosine_similarity(&embed_b) > 0.7;
                                    neighbors[i] = chunk_a.to_string();
                                    if is_neighbor {
                                        neighbors[i] += format!("\n{}", chunk_b).as_str();
                                    }
                                }
                                bar.inc(1);
                            }
                            bar.finish();

                            for chunk in neighbors {
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

                                vector.execute("INSERT INTO document (embedding, text, name) VALUES ($1, $2, $3)", 
                                        &[
                                            &Vector::from(embeddings.to_vec()),
                                            &chunk,
                                            &path.to_str().unwrap().split("/").last().expect("Failed to get document name")
                                        ])
                                        .await
                                        .expect("Failed to insert document");
                            }
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

#[derive(Database)]
#[database("pgvector")]
struct PgVector(deadpool_postgres::Pool);
