//! `lapid run <model> [inputs…]`: chat in the terminal (streaming), or run a model once on
//! its inputs: transcribe WAV files, speak text into a WAV file, embed texts.

use std::io::{BufRead, IsTerminal, Write};
use std::path::PathBuf;

use lapid::{Error, Result};
use serde_json::{Value, json};

use crate::load::{self, LoadOptions, ModelSpec};
use crate::served::{AsrParams, Capability, ChatParams, Delta, EmbedParams, GenParams, Served, TtsParams};

#[derive(Clone, Debug, Default)]
pub struct RunOptions {
    pub model: String,
    /// Prompt words (chat), WAV files (transcription), text (speech, embeddings).
    pub inputs: Vec<String>,
    /// Chat: interactive even with inputs (they are ignored then).
    pub interactive: bool,
    pub device: Option<String>,
    pub load: LoadOptions,
    pub system: Option<String>,
    pub sampling: GenParams,
    pub enable_thinking: Option<bool>,
    /// Transcript format: text, json, srt, vtt.
    pub format: Option<String>,
    /// Speech output file.
    pub output: Option<PathBuf>,
    pub voice: Option<String>,
    pub language: Option<String>,
}

fn io(e: std::io::Error) -> Error {
    Error::new("io", e.to_string())
}

pub fn run(opts: RunOptions) -> Result<()> {
    let device = crate::device(opts.device.as_deref())?;
    let spec = ModelSpec::parse(&opts.model);
    let mut model = load::load_spec(&spec, &device, &opts.load)?;
    let caps = model.meta().capabilities.clone();
    if caps.contains(&Capability::Chat) {
        chat(model.as_mut(), &opts)
    } else if caps.contains(&Capability::Transcription) {
        transcribe(model.as_mut(), &opts)
    } else if caps.contains(&Capability::Speech) {
        speak(model.as_mut(), &opts)
    } else if caps.contains(&Capability::Embedding) {
        embed(model.as_mut(), &opts)
    } else {
        Err(Error::new("unsupported_model", "this model has nothing to run"))
    }
}

struct Printer {
    color: bool,
    in_reasoning: bool,
}

impl Printer {
    fn delta(&mut self, d: Delta) -> bool {
        let mut out = std::io::stdout().lock();
        let r = match d {
            Delta::Reasoning(t) => {
                if !self.in_reasoning && self.color {
                    let _ = write!(out, "\x1b[2m");
                }
                self.in_reasoning = true;
                write!(out, "{t}")
            }
            Delta::Content(t) => {
                if self.in_reasoning {
                    let _ = writeln!(out, "{}", if self.color { "\x1b[0m" } else { "" });
                    self.in_reasoning = false;
                }
                write!(out, "{t}")
            }
        };
        r.and_then(|_| out.flush()).is_ok()
    }

    fn end(&mut self) {
        if self.in_reasoning && self.color {
            print!("\x1b[0m");
        }
        self.in_reasoning = false;
        println!();
    }
}

fn chat(model: &mut dyn Served, opts: &RunOptions) -> Result<()> {
    let color = std::io::stdout().is_terminal();
    let mut p = Printer { color, in_reasoning: false };
    let model_id = model.meta().id.clone();
    let mut history: Vec<Value> = Vec::new();
    if let Some(s) = &opts.system {
        history.push(json!({"role": "system", "content": s}));
    }
    let mut turn = |history: &mut Vec<Value>, text: &str, p: &mut Printer| -> Result<()> {
        history.push(json!({"role": "user", "content": text}));
        let req = ChatParams {
            messages: Value::Array(history.clone()),
            tools: None,
            enable_thinking: opts.enable_thinking,
            sampling: opts.sampling.clone(),
        };
        let mut answer = String::new();
        let s = model.chat(&req, &mut |d| {
            if let Delta::Content(t) = &d {
                answer.push_str(t);
            }
            p.delta(d)
        })?;
        p.end();
        if color {
            eprintln!("\x1b[2m[{} tokens, {:.1} tok/s, ttft {:.0} ms]\x1b[0m", s.completion_tokens, s.decode_tps(), s.ttft_ms);
        }
        history.push(json!({"role": "assistant", "content": answer}));
        Ok(())
    };
    if !opts.inputs.is_empty() && !opts.interactive {
        return turn(&mut history, &opts.inputs.join(" "), &mut p);
    }
    let stdin = std::io::stdin();
    let interactive = stdin.is_terminal();
    if interactive {
        eprintln!("lapid: chatting with {} (/reset clears the conversation, /exit or Ctrl-D quits)", model_id);
    }
    let base = history.len();
    loop {
        if interactive {
            print!("{}", if color { "\x1b[1m> \x1b[0m" } else { "> " });
            std::io::stdout().flush().map_err(io)?;
        }
        let mut line = String::new();
        if stdin.lock().read_line(&mut line).map_err(io)? == 0 {
            break;
        }
        let line = line.trim();
        match line {
            "" => continue,
            "/exit" | "/quit" => break,
            "/reset" => {
                history.truncate(base);
                eprintln!("lapid: conversation cleared");
                continue;
            }
            _ => {}
        }
        if let Err(e) = turn(&mut history, line, &mut p) {
            eprintln!("{}", e.to_json());
            history.pop();
        }
    }
    Ok(())
}

fn transcribe(model: &mut dyn Served, opts: &RunOptions) -> Result<()> {
    if opts.inputs.is_empty() {
        return Err(Error::new("invalid_argument", "give one or more WAV files to transcribe"));
    }
    let fmt = opts.format.as_deref().unwrap_or("text");
    for f in &opts.inputs {
        let samples = lapid_hub::audio::load_audio_16k_mono(f.as_ref())?;
        let t = model.transcribe(&samples, &AsrParams { language: opts.language.clone(), translate: false })?;
        match fmt {
            "text" => println!("{}", t.text),
            "srt" => print!("{}", crate::format::srt(&t.segments)),
            "vtt" => print!("{}", crate::format::vtt(&t.segments)),
            "json" => println!(
                "{}",
                json!({"file": f, "text": t.text, "language": t.language, "segments": t.segments, "words": t.words})
            ),
            other => return Err(Error::new("invalid_argument", format!("--format {other:?} (text, json, srt, vtt)"))),
        }
    }
    Ok(())
}

fn speak(model: &mut dyn Served, opts: &RunOptions) -> Result<()> {
    let text = opts.inputs.join(" ");
    if text.trim().is_empty() {
        return Err(Error::new("invalid_argument", "give the text to speak"));
    }
    let out = opts.output.clone().unwrap_or_else(|| PathBuf::from("speech.wav"));
    let mut pcm = Vec::new();
    let o = TtsParams { voice: opts.voice.clone(), ..Default::default() };
    model.speak(&text, &o, &mut |chunk| {
        pcm.extend_from_slice(chunk);
        true
    })?;
    let rate = model.sample_rate();
    std::fs::write(&out, crate::format::wav(&pcm, rate)).map_err(io)?;
    eprintln!("lapid: wrote {} ({:.1} s of audio)", out.display(), pcm.len() as f64 / rate.max(1) as f64);
    Ok(())
}

fn embed(model: &mut dyn Served, opts: &RunOptions) -> Result<()> {
    if opts.inputs.is_empty() {
        return Err(Error::new("invalid_argument", "give one or more texts to embed"));
    }
    let e = model.embed(&opts.inputs, &EmbedParams::default())?;
    for (t, v) in opts.inputs.iter().zip(&e.vectors) {
        println!("{}", json!({"text": t, "embedding": v}));
    }
    Ok(())
}
