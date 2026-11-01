//! Minimal Jinja2 interpreter, scoped to rendering Hugging Face chat templates exactly like
//! transformers' `apply_chat_template`: `ImmutableSandboxedEnvironment(trim_blocks=True,
//! lstrip_blocks=True)` with loop controls, `{% generation %}`, `raise_exception(msg)` and a
//! `tojson` that behaves like `json.dumps(x, ensure_ascii=False)`.
//!
//! Supported: `{{ }}` / `{% %}` / `{# #}` with `-`/`+` whitespace control; `if`/`elif`/`else`,
//! `for` (tuple unpacking, inline `if` filter, `else`, `loop.*`, `break`/`continue`), `set`
//! (plain, `ns.attr`, tuple, block), `macro`, `generation`; Python-like expressions, filters,
//! tests and dict/str methods. Not supported: template inheritance/includes, `call` blocks,
//! autoescape, `filter` blocks, `do`, `recursive` loops.

mod ast;
mod builtins;
mod eval;
mod json;
mod lexer;
mod parser;
mod value;

#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests_ref;

pub use value::Value;

/// A parsed template.
#[derive(Clone, Debug)]
pub struct Template {
    body: Vec<ast::Stmt>,
    macros: Vec<ast::MacroDef>,
}

// `Template` is stored in `Send + Sync` model structs.
const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<Template>();
    assert_send_sync::<Value>();
};

impl Template {
    /// Parse template source. Errors carry code "template".
    pub fn parse(source: &str) -> lapid::Result<Template> {
        let parsed = parser::parse(lexer::tokenize(source)?)?;
        Ok(Template { body: parsed.body, macros: parsed.macros })
    }

    /// Render with global variables. Errors carry code "template" (also for
    /// `raise_exception(msg)`, whose message is `msg`).
    pub fn render(&self, globals: &[(&str, Value)]) -> lapid::Result<String> {
        eval::Renderer::new(&self.macros, globals).render(&self.body)
    }
}
