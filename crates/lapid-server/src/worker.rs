//! One worker thread per loaded model. The model is loaded on, and never leaves, its
//! worker thread; requests are closures queued FIFO and run one at a time. Async handlers
//! await results through oneshot channels or stream them through mpsc channels; a request
//! whose client went away before it ran is skipped by its own closure.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc;

use lapid::{Device, Error, Result};

use crate::load::{self, LoadOptions};
use crate::served::{ModelMeta, Served};

type Job = Box<dyn FnOnce(&mut dyn Served) + Send>;

/// Handle to a model's worker thread (cheap to clone).
#[derive(Clone)]
pub struct Worker {
    tx: mpsc::Sender<Job>,
    meta: Arc<ModelMeta>,
    queued: Arc<AtomicUsize>,
    busy: Arc<AtomicBool>,
}

impl Worker {
    /// Spawn a thread that loads `path` on `device` and then serves jobs. Returns once the
    /// model is loaded (or failed to load).
    pub fn spawn(path: std::path::PathBuf, id: String, device: Device, opts: LoadOptions) -> Result<Worker> {
        let (tx, rx) = mpsc::channel::<Job>();
        let (ready_tx, ready_rx) = mpsc::channel::<Result<ModelMeta>>();
        let busy = Arc::new(AtomicBool::new(false));
        let queued = Arc::new(AtomicUsize::new(0));
        let (b, q) = (busy.clone(), queued.clone());
        let name = format!("lapid-model-{id}");
        std::thread::Builder::new()
            .name(name)
            .spawn(move || {
                let mut served = match load::load(&path, id, &device, &opts) {
                    Ok(s) => s,
                    Err(e) => {
                        let _ = ready_tx.send(Err(e));
                        return;
                    }
                };
                if ready_tx.send(Ok(served.meta().clone())).is_err() {
                    return;
                }
                while let Ok(job) = rx.recv() {
                    q.fetch_sub(1, Ordering::SeqCst);
                    b.store(true, Ordering::SeqCst);
                    // A panicking request must not take the model down with it.
                    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| job(served.as_mut())));
                    if r.is_err() {
                        eprintln!("lapid: a request panicked on model {}", served.meta().id);
                    }
                    b.store(false, Ordering::SeqCst);
                }
            })
            .map_err(|e| Error::new("io", format!("cannot spawn model thread: {e}")))?;
        let meta = ready_rx.recv().map_err(|_| Error::new("internal", "model thread exited while loading"))??;
        Ok(Worker { tx, meta: Arc::new(meta), queued, busy })
    }

    pub fn meta(&self) -> &ModelMeta {
        &self.meta
    }

    /// Requests waiting (not counting the running one).
    pub fn queued(&self) -> usize {
        self.queued.load(Ordering::SeqCst)
    }

    pub fn busy(&self) -> bool {
        self.busy.load(Ordering::SeqCst)
    }

    /// Queue `job`; it runs after the requests already queued.
    pub fn submit(&self, job: impl FnOnce(&mut dyn Served) + Send + 'static) -> Result<()> {
        self.queued.fetch_add(1, Ordering::SeqCst);
        self.tx.send(Box::new(job)).map_err(|_| {
            self.queued.fetch_sub(1, Ordering::SeqCst);
            Error::new("internal", "model worker stopped")
        })
    }

    /// Run `f` on the model and await its result. Skipped (never run) when the caller is
    /// dropped while queued.
    pub async fn call<R: Send + 'static>(&self, f: impl FnOnce(&mut dyn Served) -> Result<R> + Send + 'static) -> Result<R> {
        let (tx, rx) = tokio::sync::oneshot::channel();
        self.submit(move |s| {
            if tx.is_closed() {
                return;
            }
            let _ = tx.send(f(s));
        })?;
        rx.await.map_err(|_| Error::new("internal", "the request was dropped by the model worker"))?
    }
}
