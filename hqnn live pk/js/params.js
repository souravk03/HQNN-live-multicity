// Model parameter counts (POC figures). Each classical baseline is paired with its
// quantum-optimised counterpart, with the parameter reduction. These are fixed POC
// numbers; the panel just displays them and switches with the MV/UV toggle.
const MODEL_PARAMS = {
  multivariate: [
    { cls: "lstm", clsP: 5281,  q: "qlstm", qP: 2045, cmp: 60 },
    { cls: "gru",  clsP: 2473,  q: "qgru",  qP: 1195, cmp: 50 },
    { cls: "ann",  clsP: 11553, q: "hqnn",  qP: 3089, cmp: 73 },
  ],
  univariate: [
    { cls: "lstm", clsP: 3193,  q: "qlstm", qP: 1237, cmp: 73 },
    { cls: "gru",  clsP: 1265,  q: "qgru",  qP: 621,  cmp: 61 },
    { cls: "ann",  clsP: 3217,  q: "hqnn",  qP: 885,  cmp: 72 },
  ],
};

function _mpFmt(n) { try { return Number(n).toLocaleString(); } catch (e) { return String(n); } }
function _mpUP(m) { try { return mUP(m); } catch (e) { return (m || "").toUpperCase(); } }

function renderModelParams() {
  const el = document.getElementById("modelParams");
  if (!el) return;
  const mode = (typeof MODE !== "undefined" && MODE === "univariate") ? "univariate" : "multivariate";
  const rows = MODEL_PARAMS[mode] || [];
  const modeLabel = mode === "univariate" ? "Univariate" : "Multivariate";

  let h = `<table class="ptbl"><thead><tr>` +
          `<th>Pair</th><th>Classical</th><th>Quantum&#8209;optimised</th><th>Parameters saved</th></tr></thead><tbody>`;
  for (const r of rows) {
    const saved = r.clsP - r.qP;
    h += `<tr>` +
         `<td class="parch">${_mpUP(r.cls)} → <span class="qname">${_mpUP(r.q)}</span></td>` +
         `<td class="pcls"><b>${_mpFmt(r.clsP)}</b></td>` +
         `<td class="pq"><b>${_mpFmt(r.qP)}</b></td>` +
         `<td><span class="cmpbadge">▼ ${r.cmp}%</span> <span class="savedn">(−${_mpFmt(saved)})</span></td>` +
         `</tr>`;
  }
  h += `</tbody></table>`;
  el.innerHTML = h;

  const pill = document.getElementById("mpMode");
  if (pill) pill.textContent = modeLabel;
}
