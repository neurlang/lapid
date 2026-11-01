//! Declarative tensor naming and precision: a model lists name rules (upstream name pattern
//! -> our name) and role rules (our name pattern -> dtype-policy [`Role`]); the generic
//! converter (`crate::engine`) applies them.
//!
//! Patterns are literal text with two wildcards:
//! * `{x}` captures one dotted segment (one or more characters other than `.`), e.g. a
//!   layer index; the same `{x}` in the target is replaced by the captured text;
//! * `*` matches any text (possibly empty, across dots) and captures nothing.
//!
//! ```
//! use lapid_convert::mapping::{Renamed, Rule, rename};
//! const RULES: &[Rule] = &[Rule::new("blk.{i}.attn_q.{w}", "model.layers.{i}.self_attn.q_proj.{w}"), Rule::drop("rope_freqs.weight")];
//! assert_eq!(rename(RULES, "blk.3.attn_q.weight"), Renamed::To("model.layers.3.self_attn.q_proj.weight".into()));
//! assert_eq!(rename(RULES, "rope_freqs.weight"), Renamed::Drop);
//! assert_eq!(rename(RULES, "other"), Renamed::NoMatch);
//! ```

use crate::policy::Role;

/// Upstream name pattern -> our name (`to` empty = drop the tensor).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Rule {
    pub from: &'static str,
    pub to: &'static str,
}

impl Rule {
    pub const fn new(from: &'static str, to: &'static str) -> Rule {
        Rule { from, to }
    }

    /// Tensors matching `from` are not written.
    pub const fn drop(from: &'static str) -> Rule {
        Rule { from, to: "" }
    }
}

/// Our name pattern -> dtype-policy role (first matching rule wins).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct RoleRule {
    pub pattern: &'static str,
    pub role: Role,
}

impl RoleRule {
    pub const fn new(pattern: &'static str, role: Role) -> RoleRule {
        RoleRule { pattern, role }
    }
}

/// Result of [`rename`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Renamed {
    To(String),
    Drop,
    NoMatch,
}

/// Match `name` against `pattern`; returns the `{x}` captures in order.
pub fn matches<'a>(pattern: &'a str, name: &'a str) -> Option<Vec<(&'a str, &'a str)>> {
    let mut caps = Vec::new();
    if match_at(pattern, name, &mut caps) { Some(caps) } else { None }
}

fn match_at<'a>(p: &'a str, n: &'a str, caps: &mut Vec<(&'a str, &'a str)>) -> bool {
    if p.is_empty() {
        return n.is_empty();
    }
    if let Some(rest) = p.strip_prefix('*') {
        for i in (0..=n.len()).filter(|&i| n.is_char_boundary(i)) {
            let mark = caps.len();
            if match_at(rest, &n[i..], caps) {
                return true;
            }
            caps.truncate(mark);
        }
        return false;
    }
    if p.starts_with('{')
        && let Some(close) = p.find('}')
    {
        let (key, rest) = (&p[1..close], &p[close + 1..]);
        let seg_end = n.find('.').unwrap_or(n.len());
        for end in (1..=seg_end).rev().filter(|&e| n.is_char_boundary(e)) {
            let mark = caps.len();
            caps.push((key, &n[..end]));
            if match_at(rest, &n[end..], caps) {
                return true;
            }
            caps.truncate(mark);
        }
        return false;
    }
    let lit_end = p.find(['*', '{']).unwrap_or(p.len()).max(1);
    let lit = &p[..lit_end];
    match n.strip_prefix(lit) {
        Some(rest) => match_at(&p[lit_end..], rest, caps),
        None => false,
    }
}

/// `template` with every `{x}` replaced by its capture.
pub fn fill(template: &str, caps: &[(&str, &str)]) -> String {
    let mut out = template.to_string();
    for (k, v) in caps {
        out = out.replace(&format!("{{{k}}}"), v);
    }
    out
}

/// Our name of upstream tensor `name` under the first matching rule.
pub fn rename(rules: &[Rule], name: &str) -> Renamed {
    for r in rules {
        if let Some(caps) = matches(r.from, name) {
            return if r.to.is_empty() { Renamed::Drop } else { Renamed::To(fill(r.to, &caps)) };
        }
    }
    Renamed::NoMatch
}

/// Default role: biases, norms and 1-D tensors are `Param`, 2-D (and 3-D expert stacks)
/// `Linear`, everything else `Dense`.
pub fn default_role(name: &str, shape: &[usize]) -> Role {
    if name.ends_with(".bias") || name.contains("norm") || shape.len() <= 1 {
        Role::Param
    } else if matches!(shape.len(), 2) {
        Role::Linear
    } else {
        Role::Dense
    }
}

/// Role of `name` under the first matching rule, else [`default_role`].
pub fn role_of(rules: &[RoleRule], name: &str, shape: &[usize]) -> Role {
    if shape.len() <= 1 {
        return Role::Param;
    }
    rules.iter().find(|r| matches(r.pattern, name).is_some()).map_or_else(|| default_role(name, shape), |r| r.role)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn patterns() {
        assert_eq!(matches("blk.{i}.ffn_up.{w}", "blk.12.ffn_up.weight"), Some(vec![("i", "12"), ("w", "weight")]));
        assert!(matches("blk.{i}.ffn_up.{w}", "blk.12.ffn_up_exps.weight").is_none());
        assert!(matches("*norm*", "model.layers.0.post_ffn_norm.weight").is_some());
        assert!(matches("model.embed_tokens.weight", "model.embed_tokens.weight").is_some());
        assert!(matches("*.conv.conv.weight", "model.layers.3.conv.conv.weight").is_some());
        assert!(matches("{a}", "x.y").is_none());
        assert_eq!(fill("layers.{i}.{w}", &[("i", "3"), ("w", "bias")]), "layers.3.bias");
    }

    #[test]
    fn roles() {
        const R: &[RoleRule] = &[RoleRule::new("model.embed_tokens.weight", Role::LinearInt8), RoleRule::new("*.conv.weight", Role::Dense)];
        assert_eq!(role_of(R, "model.embed_tokens.weight", &[10, 4]), Role::LinearInt8);
        assert_eq!(role_of(R, "a.conv.weight", &[4, 1, 3]), Role::Dense);
        assert_eq!(role_of(R, "a.q_proj.weight", &[4, 4]), Role::Linear);
        assert_eq!(role_of(R, "a.q_norm.weight", &[4]), Role::Param);
        assert_eq!(role_of(R, "a.experts", &[2, 4, 4]), Role::Dense);
    }
}
