import { useEffect, useRef, useState } from "react";

const STYLES = [
  ["Comic Book", "🦸"], ["Manga", "🌸"], ["Storybook", "🎨"],
  ["Noir", "🕵️"], ["3D Animation", "🧸"], ["Retro Sci-Fi", "🚀"],
];
const EXAMPLES = [
  "Mira, a girl with short red hair and a yellow raincoat, finds a glowing key in a rainy alley. It opens a hidden door in an old bookstore. Inside, a tiny dragon is asleep on a pile of maps. They become friends and fly over the city at sunrise.",
  "Captain Juno, a tall astronaut with a silver braid, lands on a silent red planet. Her robot Bolt, a small round yellow bot, detects a signal. They follow it into a crystal cave and discover a garden growing in the dark.",
];
const STEPS = [["planning", "Writing the script"], ["drawing", "Drawing the panels"], ["done", "Ready to read"]];

export default function App() {
  const [story, setStory] = useState("");
  const [style, setStyle] = useState("Comic Book");
  const [count, setCount] = useState(6);
  const [job, setJob] = useState(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const timer = useRef();
  useEffect(() => () => clearInterval(timer.current), []);

  const poll = (id) => {
    clearInterval(timer.current);
    timer.current = setInterval(async () => {
      try {
        const j = await (await fetch(`/api/stories/${id}`)).json();
        setJob(j);
        if (j.status === "done" || j.status === "error") { clearInterval(timer.current); setBusy(false); }
      } catch { /* retry on next tick */ }
    }, 2000);
  };

  const create = async () => {
    setError(""); setBusy(true);
    try {
      const r = await fetch("/api/stories", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ story, style, panels: count }),
      });
      if (!r.ok) { const d = (await r.json()).detail; throw new Error(Array.isArray(d) ? d[0].msg : d || "Could not start"); }
      const { id } = await r.json();
      setJob({ id, status: "planning", characters: [], panels: [] });
      poll(id);
    } catch (e) { setError(e.message); setBusy(false); }
  };

  const reset = () => { clearInterval(timer.current); setJob(null); setBusy(false); setError(""); };
  const stepIdx = job ? STEPS.findIndex(([k]) => k === job.status) : -1;
  const slots = job?.panels?.length ? job.panels : Array.from({ length: count }, () => ({}));

  return (
    <div className="app">
      <header>
        <div className="logo">COMIC<span>FORGE</span></div>
        {job && <button className="ghost" onClick={reset}>＋ New story</button>}
      </header>

      {!job ? (
        <main className="hero">
          <h1>Turn any story into a <em>comic</em>.</h1>
          <p className="sub">Write a few lines. AI plans the panels, designs your characters once, and keeps them looking the same on every page.</p>

          <section className="card">
            <label>Your story</label>
            <textarea value={story} onChange={(e) => setStory(e.target.value)} rows={6}
              placeholder="Describe your characters and what happens to them…" />
            <div className="row between">
              <div className="examples">
                {EXAMPLES.map((ex, i) => <button key={i} className="chip" onClick={() => setStory(ex)}>✨ Example {i + 1}</button>)}
              </div>
              <span className="muted">{story.length}/3000</span>
            </div>

            <label>Art style</label>
            <div className="styles">
              {STYLES.map(([name, icon]) => (
                <button key={name} className={"style " + (style === name ? "on" : "")} onClick={() => setStyle(name)}>
                  <span>{icon}</span>{name}
                </button>
              ))}
            </div>

            <label>Panels: <b>{count}</b></label>
            <input type="range" min="2" max="8" value={count} onChange={(e) => setCount(+e.target.value)} />

            {error && <div className="error">{error}</div>}
            <button className="cta" disabled={busy || story.trim().length < 20} onClick={create}>
              {busy ? "Starting…" : "Forge my comic →"}
            </button>
          </section>
        </main>
      ) : (
        <main className="studio">
          <div className="steps">
            {STEPS.map(([k, label], i) => (
              <div key={k} className={"step " + (i < stepIdx || job.status === "done" ? "ok " : "") + (i === stepIdx ? "now" : "")}>
                <i>{i < stepIdx || job.status === "done" ? "✓" : i + 1}</i>{label}
              </div>
            ))}
          </div>

          {job.status === "error" && <div className="error">Something went wrong: {job.error}</div>}

          {job.characters.length > 0 && (
            <section>
              <h2>The cast</h2>
              <div className="cast">
                {job.characters.map((c) => (
                  <div className="char" key={c.name}>
                    <div className="avatar">{c.name[0]}</div>
                    <div><b>{c.name}</b><p>{c.description}</p></div>
                  </div>
                ))}
              </div>
            </section>
          )}

          <section>
            <div className="row between">
              <h2>Your comic</h2>
              {job.status === "done" && <a className="cta small" href={`/api/stories/${job.id}/pdf`}>⬇ Download PDF</a>}
            </div>
            <div className="grid">
              {slots.map((p, i) => (
                <figure className="panel" key={i} style={{ "--tilt": i % 2 ? "0.4deg" : "-0.4deg" }}>
                  {p.image ? <img src={p.image} alt={p.caption} /> : <div className="skeleton"><span>{i + 1}</span></div>}
                  {p.caption && <figcaption>{p.caption}</figcaption>}
                </figure>
              ))}
            </div>
          </section>
        </main>
      )}
      <footer>Built with FastAPI · React · FLUX.1 · consistency via fixed character sheets + shared seed</footer>
    </div>
  );
}
