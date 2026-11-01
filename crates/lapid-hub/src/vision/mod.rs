//! Shared pieces of the vision-language models: chat messages with image parts
//! ([`Message`]), the generic ViT / SigLIP-style encoder behind GGUF `clip` / mmproj files and
//! its preprocessing settings. Images, patching, 2-D RoPE and pooling are `lapid_nn::vision`.

pub mod encoder;
pub mod processor;

pub use lapid_nn::vision::{Filter, Image};

use crate::runtime::jinja;
pub use encoder::{VisionConfig, VisionEncoder};
pub use processor::{ImageProcessor, Patches};

/// One part of a message's content.
#[derive(Clone, Debug)]
pub enum Part {
    Text(String),
    Image(Image),
}

/// A chat message whose content may mix text and images (`role`: "system" | "user" |
/// "assistant"). Rendered for chat templates as HF-style content parts
/// (`{"type": "image"}` / `{"type": "text", "text": ...}`).
#[derive(Clone, Debug, Default)]
pub struct Message {
    pub role: String,
    pub content: Vec<Part>,
}

impl Message {
    pub fn new(role: &str, content: Vec<Part>) -> Message {
        Message { role: role.to_string(), content }
    }

    pub fn text(role: &str, text: impl Into<String>) -> Message {
        Message::new(role, vec![Part::Text(text.into())])
    }

    pub fn system(text: impl Into<String>) -> Message {
        Message::text("system", text)
    }

    pub fn user(text: impl Into<String>) -> Message {
        Message::text("user", text)
    }

    pub fn assistant(text: impl Into<String>) -> Message {
        Message::text("assistant", text)
    }

    /// A user turn with the images first, then the text (the usual HF example layout).
    pub fn user_with_images(images: Vec<Image>, text: impl Into<String>) -> Message {
        let mut content: Vec<Part> = images.into_iter().map(Part::Image).collect();
        content.push(Part::Text(text.into()));
        Message::new("user", content)
    }

    /// The images of this message, in order.
    pub fn images(&self) -> impl Iterator<Item = &Image> {
        self.content.iter().filter_map(|p| match p {
            Part::Image(i) => Some(i),
            Part::Text(_) => None,
        })
    }

    /// Template value: `content` is a plain string for text-only messages, else a list of parts.
    pub fn to_value(&self) -> jinja::Value {
        let text_only = self.content.iter().all(|p| matches!(p, Part::Text(_)));
        let content = if text_only {
            let s: String = self
                .content
                .iter()
                .filter_map(|p| match p {
                    Part::Text(t) => Some(t.as_str()),
                    Part::Image(_) => None,
                })
                .collect();
            jinja::Value::str(s)
        } else {
            jinja::Value::List(
                self.content
                    .iter()
                    .map(|p| match p {
                        Part::Text(t) => jinja::Value::map(vec![
                            ("type".to_string(), jinja::Value::str("text")),
                            ("text".to_string(), jinja::Value::str(t.clone())),
                        ]),
                        Part::Image(_) => jinja::Value::map(vec![("type".to_string(), jinja::Value::str("image"))]),
                    })
                    .collect(),
            )
        };
        jinja::Value::map(vec![("role".to_string(), jinja::Value::str(&self.role)), ("content".to_string(), content)])
    }
}

/// All images of `messages` in order (they map to the template's image placeholders in order).
pub fn collect_images(messages: &[Message]) -> Vec<&Image> {
    messages.iter().flat_map(Message::images).collect()
}
