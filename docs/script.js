const revealObserver = new IntersectionObserver(
  (entries) => {
    for (const entry of entries) {
      if (entry.isIntersecting) {
        entry.target.classList.add("visible");
        revealObserver.unobserve(entry.target);
      }
    }
  },
  { threshold: 0.08 }
);

document.querySelectorAll(".reveal").forEach((element) => revealObserver.observe(element));

const worldStage = document.querySelector(".world-stage");
const worldSteps = [...document.querySelectorAll(".world-step")];
const latentLabel = document.querySelector("#latent-label");
const latentSymbol = document.querySelector("#latent-symbol");
const observationStream = document.querySelector("#observation-stream");
const beliefBars = document.querySelector("#belief-bars");
let activeWorld = "coin";
let worldTransition;

function renderWorld(step) {
  if (!worldStage || step.dataset.theme === activeWorld) return;

  activeWorld = step.dataset.theme;
  worldSteps.forEach((item) => item.classList.toggle("active", item === step));
  worldStage.classList.add("changing");
  window.clearTimeout(worldTransition);

  worldTransition = window.setTimeout(() => {
    worldStage.dataset.theme = step.dataset.theme;
    latentLabel.textContent = step.dataset.latent;
    latentSymbol.textContent = step.dataset.symbol;

    const observations = step.dataset.observations.split(",");
    observationStream.replaceChildren(
      ...observations.map((value) => {
        const token = document.createElement("b");
        token.textContent = value;
        return token;
      })
    );

    const bars = step.dataset.bars.split(",");
    beliefBars.replaceChildren(
      ...bars.map((height) => {
        const bar = document.createElement("i");
        bar.style.setProperty("--h", `${height}%`);
        return bar;
      })
    );

    worldStage.classList.remove("changing");
  }, 180);
}

const worldObserver = new IntersectionObserver(
  (entries) => {
    const visible = entries
      .filter((entry) => entry.isIntersecting)
      .sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];
    if (visible) renderWorld(visible.target);
  },
  { rootMargin: "-28% 0px -42%", threshold: [0, 0.25, 0.5] }
);

worldSteps.forEach((step) => worldObserver.observe(step));

worldStage?.addEventListener("pointermove", (event) => {
  const bounds = worldStage.getBoundingClientRect();
  const x = ((event.clientX - bounds.left) / bounds.width) * 100;
  const y = ((event.clientY - bounds.top) / bounds.height) * 100;
  worldStage.style.setProperty("--mx", `${x}%`);
  worldStage.style.setProperty("--my", `${y}%`);
});

worldStage?.addEventListener("pointerleave", () => {
  worldStage.style.setProperty("--mx", "50%");
  worldStage.style.setProperty("--my", "45%");
});

const copyButton = document.querySelector("#copy-citation");
const citation = document.querySelector("#bibtex");

copyButton?.addEventListener("click", async () => {
  try {
    await navigator.clipboard.writeText(citation.textContent);
    copyButton.textContent = "Copied";
    window.setTimeout(() => {
      copyButton.textContent = "Copy citation";
    }, 1800);
  } catch {
    const range = document.createRange();
    range.selectNodeContents(citation);
    window.getSelection().removeAllRanges();
    window.getSelection().addRange(range);
    copyButton.textContent = "Selected";
  }
});
