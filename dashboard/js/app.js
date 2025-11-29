let mapContainer;
let infoContainer;
let resetButton;
const tabButtons = document.querySelectorAll(".tab-button");
const tabContent = document.getElementById("tab-content");

let mapInitialized = false;
let currentRegion = null;
let currentTab = "overview";

function destroyMapListeners() {
  if (mapContainer && mapInitialized && mapContainer.removeAllListeners) {
    mapContainer.removeAllListeners("plotly_click");
  }
  mapInitialized = false;
}

function setActiveTab(tabId) {
  tabButtons.forEach((button) => {
    const isActive = button.dataset.tab === tabId;
    button.classList.toggle("active", isActive);
    button.setAttribute("aria-selected", String(isActive));
  });
}

async function loadTabContent(tabId) {
  const response = await fetch(`/tabs/${tabId}.html`);
  const html = await response.text();
  tabContent.innerHTML = html;
  currentTab = tabId;

  if (tabId === "overview") {
    mapContainer = document.getElementById("world-map");
    infoContainer = document.getElementById("region-info");
    resetButton = document.getElementById("reset-btn");
    attachOverviewHandlers();
    await Promise.all([
      renderMap(currentRegion),
      updateRegionInfo(currentRegion),
      updateOverviewSummary(),
    ]);
    updateComplianceBlock();
    renderForecastChart();
    renderTrendChart();
  } else {
    destroyMapListeners();
  }
}

async function fetchFigure(region = null) {
  const query = region ? `?selected=${encodeURIComponent(region)}` : "";
  const response = await fetch(`/api/world-map${query}`);
  const payload = await response.json();
  return payload.figure;
}

async function fetchRegionInfo(region = null) {
  const query = region ? `?region=${encodeURIComponent(region)}` : "";
  const response = await fetch(`/api/region-info${query}`);
  return response.json();
}

async function fetchOverviewSummary() {
  const response = await fetch("/api/overview-summary");
  return response.json();
}

function renderRegionInfo(info) {
  if (!infoContainer) return;

  if (info.default) {
    infoContainer.innerHTML = `
      <h2>Information</h2>
      <p>${info.description}</p>
      <ul>
        ${info.highlights.map((item) => `<li>${item}</li>`).join("")}
      </ul>
    `;
    return;
  }

  const countryList = info.countries
    .slice(0, 20)
    .map((country) => `<li>${country}</li>`)
    .join("");

  infoContainer.innerHTML = `
    <h2>${info.region}</h2>
    <p>${info.description}</p>
    <p><strong>${info.countryCount}</strong> countries in this view.</p>
    <h4>Countries</h4>
    <ul>${countryList}</ul>
    ${
      info.countries.length > 20
        ? `<small>…and more.</small>`
        : "<small>&nbsp;</small>"
    }
  `;
}

async function renderMap(region = null) {
  if (!mapContainer) return;
  const figure = await fetchFigure(region);
  const config = { responsive: true, displayModeBar: false };

  if (!mapInitialized) {
    Plotly.newPlot(mapContainer, figure.data, figure.layout, config).then(() => {
      mapContainer.on("plotly_click", handleRegionClick);
    });
    mapInitialized = true;
  } else {
    Plotly.react(mapContainer, figure.data, figure.layout, config);
  }
}

async function updateRegionInfo(region = null) {
  if (!infoContainer) return;
  const info = await fetchRegionInfo(region);
  renderRegionInfo(info);
}

async function updateOverviewSummary() {
  const riskEl = document.getElementById("risk-score-value");
  const suppliersEl = document.getElementById("total-suppliers-value");
  const donutEl = document.getElementById("risk-donut-chart");
  if (!riskEl || !suppliersEl || !donutEl) return;

  const summary = await fetchOverviewSummary();
  riskEl.textContent = `${summary.riskScore.toFixed(1)} / 100`;
  suppliersEl.textContent = summary.totalSuppliers.toLocaleString();

  const config = { displayModeBar: false, responsive: true };
  Plotly.react(donutEl, summary.donut.data, summary.donut.layout, config);
}

function updateComplianceBlock() {
  const complianceEl = document.getElementById("compliance-value");
  if (!complianceEl) return;
  complianceEl.textContent = "92% Compliant";
}

function renderForecastChart() {
  const node = document.getElementById("forecast-chart");
  if (!node) return;
  const figure = {
    data: [
      {
        values: [14, 48, 38],
        labels: ["High", "Medium", "Low"],
        type: "pie",
        hole: 0.55,
        marker: {
          colors: ["#fa7066", "#fcaf56", "#adcf7a"],
        },
      },
    ],
    layout: {
      margin: { t: 10, b: 10, l: 10, r: 10 },
      showlegend: false,
    },
  };
  Plotly.react(node, figure.data, figure.layout, { displayModeBar: false, responsive: true });
}

function renderTrendChart() {
  const node = document.getElementById("trend-chart");
  if (!node) return;
  const years = [2020, 2021, 2022, 2023, 2024, 2025];
  const series = [
    { name: "Carbon Emission Intensity", color: "#1abc9c", values: [12, 13, 14, 16, 18, 21] },
    { name: "GHG Scopes 1", color: "#3498db", values: [8, 9, 11, 12, 13, 14] },
    { name: "GHG Scopes 2", color: "#9b59b6", values: [5, 6, 6, 7, 8, 9] },
    { name: "Renewable Energy Usage", color: "#f39c12", values: [3, 4, 5, 6, 7, 8] },
  ];

  const data = series.map((serie) => ({
    x: years,
    y: serie.values,
    mode: "lines",
    name: serie.name,
    line: { color: serie.color, width: 3 },
  }));

  const layout = {
    margin: { t: 20, r: 10, b: 30, l: 40 },
    legend: { orientation: "h", y: -0.2 },
    xaxis: { title: "", gridcolor: "#edf1fa" },
    yaxis: { title: "", gridcolor: "#edf1fa" },
  };

  Plotly.react(node, data, layout, { displayModeBar: false, responsive: true });
}

async function handleRegionClick(event) {
  if (!event || !event.points || !event.points.length) return;
  const region = event.points[0].customdata?.[0];
  if (!region) return;

  currentRegion = region;
  await Promise.all([renderMap(region), updateRegionInfo(region)]);
}

async function resetMap() {
  currentRegion = null;
  await Promise.all([renderMap(), updateRegionInfo()]);
}

function attachOverviewHandlers() {
  destroyMapListeners();
  mapInitialized = false;
  if (resetButton) {
    resetButton.addEventListener("click", resetMap);
  }
}

tabButtons.forEach((button) => {
  button.addEventListener("click", () => {
    const tabId = button.dataset.tab;
    if (!tabId) return;
    setActiveTab(tabId);
    loadTabContent(tabId);
  });
});

window.addEventListener("DOMContentLoaded", () => {
  setActiveTab("overview");
  loadTabContent("overview");
});