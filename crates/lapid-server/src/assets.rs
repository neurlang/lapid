//! The embedded web UI (`ui/dist`, built with `pnpm build`; not committed, see build.rs).
//! Compressible assets are embedded only as `.gz` (smaller binary): sent as is when the client
//! accepts gzip, decompressed on the fly otherwise. Unknown non-API paths fall back to
//! `index.html` (client-side routing).

use axum::http::{HeaderMap, HeaderValue, StatusCode, Uri, header};
use axum::response::{IntoResponse, Response};
use rust_embed::Embed;

use crate::api::ApiError;

#[derive(Embed)]
#[folder = "ui/dist"]
struct Dist;

fn accepts(headers: &HeaderMap, encoding: &str) -> bool {
    headers
        .get_all(header::ACCEPT_ENCODING)
        .iter()
        .filter_map(|v| v.to_str().ok())
        .flat_map(|v| v.split(','))
        .any(|e| e.split(';').next().is_some_and(|n| n.trim() == encoding))
}

fn file_response(path: &str, headers: &HeaderMap) -> Option<Response> {
    let mime = mime_guess::from_path(path).first_or_octet_stream();
    let cache = if path.starts_with("assets/") { "public, max-age=31536000, immutable" } else { "no-cache" };
    let (body, gzipped) = match Dist::get(path) {
        Some(f) => (f.data.into_owned(), false),
        None => {
            let gz = Dist::get(&format!("{path}.gz"))?.data.into_owned();
            if accepts(headers, "gzip") {
                (gz, true)
            } else {
                let mut out = Vec::new();
                std::io::Read::read_to_end(&mut flate2::read::GzDecoder::new(&gz[..]), &mut out).ok()?;
                (out, false)
            }
        }
    };
    let mut res = (StatusCode::OK, body).into_response();
    let h = res.headers_mut();
    if let Ok(v) = HeaderValue::from_str(mime.as_ref()) {
        h.insert(header::CONTENT_TYPE, v);
    }
    h.insert(header::CACHE_CONTROL, HeaderValue::from_static(cache));
    h.insert(header::VARY, HeaderValue::from_static("accept-encoding"));
    if gzipped {
        h.insert(header::CONTENT_ENCODING, HeaderValue::from_static("gzip"));
    }
    Some(res)
}

pub async fn serve_asset(uri: Uri, headers: HeaderMap) -> Response {
    let path = uri.path().trim_start_matches('/');
    if path.starts_with("v1/") || path.starts_with("api/") {
        return ApiError::not_found("not_found", format!("no route for {}", uri.path())).into_response();
    }
    let path = if path.is_empty() { "index.html" } else { path };
    if let Some(r) = file_response(path, &headers) {
        return r;
    }
    // Missing hashed assets are real 404s; everything else is a client-side route.
    if path.starts_with("assets/") {
        return (StatusCode::NOT_FOUND, "not found").into_response();
    }
    file_response("index.html", &headers)
        .unwrap_or_else(|| (StatusCode::NOT_FOUND, "the web UI is not built (crates/lapid-server/ui: pnpm build)").into_response())
}
