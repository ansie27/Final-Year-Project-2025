let currentRegion = null;
let currentTab = "overview";
let mapInitialized = false;

// DOM element references
let mapContainer;
let resetButton;
let regionFilter;
let detailedSearchInput;
let detailedSearchButton;

const tabButtons = document.querySelectorAll(".tab-button");
const tabContent = document.getElementById("tab-content");

function setActiveTab(tabId) {
  tabButtons.forEach((button) => {
    const isActive = button.dataset.tab === tabId;
    button.classList.toggle("active", isActive);
    button.setAttribute("aria-selected", String(isActive));
  });
}

function hexToRgba(hex, alpha) {
  const sanitized = hex.replace("#", "");
  const bigint = parseInt(sanitized, 16);
  const r = (bigint >> 16) & 255;
  const g = (bigint >> 8) & 255;
  const b = bigint & 255;
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

async function loadTabContent(tabId) {
  const response = await fetch(`/tabs/${tabId}.html`);
  const html = await response.text();
  tabContent.innerHTML = html;
  currentTab = tabId;

  if (tabId === "overview") {
    initOverviewTab();
  } else if (tabId === "detailed") {
    initDetailedTab();
  } else if (tabId === "metrics") {
    // Future: initialize metrics view
  }
}

async function initOverviewTab() {
  mapContainer = document.getElementById("world-map");
  resetButton = document.getElementById("reset-btn");
  regionFilter = document.getElementById("region-filter");

  attachOverviewHandlers();

  await populateRegionFilter();
  const region = getSelectedRegion();

  await Promise.all([
    renderMap(currentRegion),
    updateOverviewSummary(region),
    loadTopCommodities(region),
    loadTopMovers(region),
    renderForecastChart(),
    renderCarbonTrajectoryChart(),
    renderTrendChart(),
  ]);
}

function getSelectedRegion() {
  if (!regionFilter) return "All";
  return regionFilter.value || "All";
}

async function fetchMapData(region = null) {
  const query = region ? `?selected=${encodeURIComponent(region)}` : "";
  const response = await fetch(`/api/map-data${query}`);
  return response.json();
}

async function fetchRegionInfo(region = null) {
  const query = region ? `?region=${encodeURIComponent(region)}` : "";
  const response = await fetch(`/api/region-info${query}`);
  return response.json();
}

async function fetchOverviewSummary(region = null) {
  const query =
    region && region !== "All" ? `?region=${encodeURIComponent(region)}` : "";
  const response = await fetch(`/api/overview-summary${query}`);
  return response.json();
}

async function fetchTopCommodities(region = "All") {
  const query =
    region && region !== "All" ? `?region=${encodeURIComponent(region)}` : "";
  const response = await fetch(`/api/top-commodities${query}`);
  return response.json();
}

async function fetchTopMovers(region = "All") {
  const query =
    region && region !== "All" ? `?region=${encodeURIComponent(region)}` : "";
  const response = await fetch(`/api/top-movers${query}`);
  return response.json();
}

async function fetchRegionOptions() {
  const response = await fetch("/api/regions");
  return response.json();
}

async function fetchForecastData() {
  const response = await fetch("/api/forecast-data");
  return response.json();
}

async function fetchCarbonTrajectory() {
  const response = await fetch("/api/carbon-trajectory");
  return response.json();
}

async function fetchTrendData() {
  const response = await fetch("/api/trend-data");
  return response.json();
}

async function fetchSupplierProfile(query = "") {
  const qs = query ? `?supplier=${encodeURIComponent(query)}` : "";
  const response = await fetch(`/api/supplier-profile${qs}`);
  if (!response.ok) {
    throw new Error("Supplier not found");
  }
  return response.json();
}

async function renderMap(region = null) {
  if (!mapContainer) return;

  const mapData = await fetchMapData(region);

  const data = mapData.category_order
    .map((category) => {
      const indices = mapData.display_regions
        .map((value, idx) => (value === category ? idx : -1))
        .filter((idx) => idx !== -1);

      if (!indices.length) {
        return null;
      }

      const color = mapData.color_map[category] || "#d9d9d9";
      const opacity = region && category !== region ? 0.3 : 1.0;
      const name =
        category === "Other Regions" ? "Other Regions" : category;

      return {
        type: "choropleth",
        locations: indices.map((idx) => mapData.locations[idx]),
        z: indices.map(() => 1),
        text: indices.map((idx) => mapData.countries[idx]),
        customdata: indices.map((idx) => mapData.regions[idx]),
        hovertemplate:
          "<b>%{text}</b><br>Region: %{customdata}<extra></extra>",
        colorscale: [
          [0, color],
          [1, color],
        ],
        showscale: false,
        name,
        showlegend: true,
        marker: {
          opacity,
          line: {
            color: "#ffffff",
            width: 0.5,
          },
        },
      };
    })
    .filter(Boolean);

  const layout = {
    geo: {
      projection: { type: "natural earth" },
      showcountries: true,
      showcoastlines: true,
      showland: true,
      landcolor: "#F5F5F5",
      coastlinecolor: "#cccccc",
      countrycolor: "#dddddd",
    },
    margin: { l: 0, r: 0, t: 40, b: 0 },
    title: {
      text: mapData.selected_region
        ? `${mapData.selected_region} Region`
        : "Global Regions Overview",
      font: { size: 20 },
      x: 0.01,
      xanchor: "left",
    },
    height: 420,
    legend: { // World map legend is here
      orientation: "h",
      y: -0.05,
      x: 0.5,
      xanchor: "center",
      bgcolor: "rgba(255, 255, 255, 0.9)",
      bordercolor: "#dfe3ef",
      borderwidth: 1,
      font: { size: 12 },
    },
  };

  const config = {
    responsive: true,
    displayModeBar: false,
  };

  if (!mapInitialized) {
    Plotly.newPlot(mapContainer, data, layout, config).then(() => {
      mapContainer.on("plotly_click", handleMapClick);
      mapInitialized = true;
    });
  } else {
    Plotly.react(mapContainer, data, layout, config);
  }
}

async function handleMapClick(event) {
  if (!event || !event.points || !event.points.length) return;

  const region = event.points[0].customdata;
  if (!region) return;

  currentRegion = region;
  await Promise.all([renderMap(region), updateOverviewSummary(region)]);
}

function destroyMapListeners() {
  if (mapContainer && mapInitialized && mapContainer.removeAllListeners) {
    mapContainer.removeAllListeners("plotly_click");
  }
  mapInitialized = false;
}

async function resetMap() {
  currentRegion = null;
  await Promise.all([renderMap(), updateOverviewSummary(getSelectedRegion())]);
}

async function initDetailedTab() {
  detailedSearchInput = document.getElementById("supplier-search");
  detailedSearchButton = document.getElementById("supplier-search-btn");

  const handleSearch = async () => {
    const value = detailedSearchInput?.value?.trim();
    await loadSupplierProfile(value || "");
  };

  if (detailedSearchButton) {
    detailedSearchButton.addEventListener("click", handleSearch);
  }
  if (detailedSearchInput) {
    detailedSearchInput.addEventListener("keydown", (event) => {
      if (event.key === "Enter") {
        event.preventDefault();
        handleSearch();
      }
    });
  }

  await loadSupplierProfile();
}

async function loadSupplierProfile(query = "") {
  try {
    const profile = await fetchSupplierProfile(query);
    populateSupplierProfile(profile);
    renderDetailedRiskTrend(profile.risk_trend);
    renderEsgBreakdown(profile.esg_breakdown);
    renderReliability(profile.reliability);
    renderRiskDrivers(profile.risk_drivers);
  } catch (error) {
    console.error(error);
  }
}

async function populateRegionFilter() {
  if (!regionFilter) return;

  const data = await fetchRegionOptions();
  const regions = Array.isArray(data.regions) ? data.regions : [];
  const options = ["All", ...regions];

  regionFilter.innerHTML = options
    .map(
      (region) => `<option value="${region}">${region.toUpperCase()}</option>`
    )
    .join("");

  regionFilter.removeEventListener("change", handleRegionFilterChange);
  regionFilter.addEventListener("change", handleRegionFilterChange);
}

async function handleRegionFilterChange() {
  const region = getSelectedRegion();
  await Promise.all([
    updateOverviewSummary(region),
    loadTopCommodities(region),
    loadTopMovers(region),
  ]);
}

async function updateOverviewSummary(regionOverride = null) {
  const region = regionOverride ?? currentRegion ?? getSelectedRegion();
  const summary = await fetchOverviewSummary(region);

  // Update risk score
  const riskEl = document.getElementById("risk-score-value");
  if (riskEl) {
    riskEl.textContent = `${summary.risk_score.toFixed(1)} / 100`;
  }

  // Update total suppliers
  const suppliersEl = document.getElementById("total-suppliers-value");
  if (suppliersEl) {
    suppliersEl.textContent = summary.total_suppliers.toLocaleString();
  }

  // Update compliance
  const complianceEl = document.getElementById("compliance-value");
  if (complianceEl) {
    complianceEl.textContent = `${summary.compliance_rate}% Compliant`;
  }

  // Render risk donut chart
  renderRiskDonut(summary.risk_distribution);
}

function renderRiskDonut(distribution) {
  const donutEl = document.getElementById("risk-donut-chart");
  if (!donutEl) return;

  const data = [
    {
      values: distribution.values,
      labels: distribution.labels,
      type: "pie",
      hole: 0.55,
      marker: {
        colors: distribution.colors,
      },
      textinfo: "label+percent",
      textposition: "inside",
    },
  ];

  const layout = {
    margin: { t: 30, b: 10, l: 10, r: 10 },
    showlegend: false,
    title: {
      text: "Supplier-Commodity Risk Mix",
      font: { size: 14 },
    },
  };

  const config = {
    displayModeBar: false,
    responsive: true,
  };

  Plotly.react(donutEl, data, layout, config);
}

async function loadTopCommodities(region = "All") {
  const data = await fetchTopCommodities(region);
  const tables = document.querySelectorAll(".table-block table");
  
  if (!tables || tables.length === 0) return;
  
  const tbody = tables[0].querySelector("tbody");
  if (!tbody) return;

  tbody.innerHTML = data.commodities
    .map(
      (c) => `
    <tr>
      <td>${c.name}</td>
      <td>${c.sustainability}</td>
      <td>${c.ghg_score.toFixed(2)}</td>
      <td>${c.cost.toFixed(1)}</td>
    </tr>
  `
    )
    .join("");
}

async function loadTopMovers(region = "All") {
  const data = await fetchTopMovers(region);
  const tables = document.querySelectorAll(".table-block table");

  if (!tables || tables.length < 2) return;

  const tbody = tables[1].querySelector("tbody");
  if (!tbody) return;

  tbody.innerHTML = data.movers
    .map(
      (m) => `
    <tr>
      <td>${m.name}</td>
      <td>${m.country ?? "N/A"}</td>
      <td>${m.resilience.toFixed(2)}</td>
      <td>${m.risk_delta.toFixed(2)}</td>
    </tr>
  `
    )
    .join("");
}

async function renderForecastChart() {
  const node = document.getElementById("forecast-chart");
  if (!node) return;

  const forecastData = await fetchForecastData();

  const data = [
    {
      values: forecastData.values,
      labels: forecastData.labels,
      type: "pie",
      hole: 0.55,
      marker: {
        colors: forecastData.colors,
      },
      textinfo: "label+percent",
      textposition: "inside",
    },
  ];

  const layout = {
    margin: { t: 10, b: 10, l: 10, r: 10 },
    showlegend: false,
  };

  const config = {
    displayModeBar: false,
    responsive: true,
  };

  Plotly.react(node, data, layout, config);
}

async function renderCarbonTrajectoryChart() {
  const node = document.getElementById("carbon-trajectory-chart");
  if (!node) return;

  const payload = await fetchCarbonTrajectory();
  const data = [
    {
      x: payload.months,
      y: payload.values,
      type: "bar",
      marker: {
        color: "#2e86ab",
        line: { color: "#1f618d", width: 1 },
      },
    },
  ];

  const layout = {
    margin: { t: 10, r: 10, b: 40, l: 50 },
    yaxis: {
      title: "Projected Intensity (index)",
      gridcolor: "#edf1fa",
      range: [0, 110],
    },
    xaxis: { title: "Months Ahead", gridcolor: "#edf1fa" },
    height: 220,
  };

  const config = {
    displayModeBar: false,
    responsive: true,
  };

  Plotly.react(node, data, layout, config);
}

async function renderTrendChart() {
  const node = document.getElementById("trend-chart");
  if (!node) return;

  const trendData = await fetchTrendData();

  const traces = [];

  (trendData.series || []).forEach((serie) => {
    if (serie.actual?.x?.length && serie.actual?.y?.length) {
      traces.push({
        x: serie.actual.x,
        y: serie.actual.y,
        mode: "lines",
        name: `${serie.name} (Actual)`,
        line: {
          color: serie.color,
          width: 3,
        },
      });
    }

    if (serie.forecast?.x?.length && serie.forecast?.y?.length) {
      const lastActualYear =
        serie.actual?.x?.length > 0
          ? serie.actual.x[serie.actual.x.length - 1]
          : null;
      const lastActualValue =
        serie.actual?.y?.length > 0
          ? serie.actual.y[serie.actual.y.length - 1]
          : null;

      const forecastX =
        lastActualYear !== null
          ? [lastActualYear, ...serie.forecast.x]
          : serie.forecast.x;
      const forecastY =
        lastActualValue !== null
          ? [lastActualValue, ...serie.forecast.y]
          : serie.forecast.y;

      traces.push({
        x: forecastX,
        y: forecastY,
        mode: "lines",
        name: `${serie.name} Forecast`,
        line: {
          color: serie.color,
          width: 3,
          dash: "dot",
        },
      });
    }

    if (
      serie.confidence?.x?.length &&
      serie.confidence.upper?.length === serie.confidence.x.length &&
      serie.confidence.lower?.length === serie.confidence.x.length
    ) {
      const xVals = serie.confidence.x;
      traces.push({
        x: [...xVals, ...xVals.slice().reverse()],
        y: [
          ...serie.confidence.upper,
          ...serie.confidence.lower.slice().reverse(),
        ],
        fill: "toself",
        fillcolor: hexToRgba(serie.color, 0.15),
        line: { color: "transparent" },
        hoverinfo: "skip",
        showlegend: false,
      });
    }
  });

  const layout = {
    margin: { t: 20, r: 10, b: 30, l: 40 },
    legend: {
      orientation: "h",
      y: -0.25,
    },
    xaxis: {
      title: "Year",
      gridcolor: "#edf1fa",
    },
    yaxis: {
      title: "Score (0-100)",
      range: [0, 100],
      gridcolor: "#edf1fa",
    },
  };

  const config = {
    displayModeBar: false,
    responsive: true,
  };

  Plotly.react(node, traces, layout, config);
}

function populateSupplierProfile(profile) {
  const setText = (id, value) => {
    const el = document.getElementById(id);
    if (el) el.textContent = value ?? "--";
  };

  setText("supplier-name", profile.supplier ?? "--");
  setText("supplier-id", profile.sc_id ?? "--");
  setText(
    "supplier-meta",
    `${profile.industry || "Industry"} · ${profile.tier || "Tier"} · ${
      profile.employees?.toLocaleString?.() || "--"
    } employees`
  );
  setText("supplier-hq", profile.country ?? "--");
  setText("supplier-commodity", profile.commodity ?? "--");
  setText("supplier-certifications", profile.certifications || "None");

  setText("supplier-risk-index", profile.risk_index ?? "--");
  setText("supplier-classification", profile.classification ?? "--");
  setText("supplier-esg-score", profile.metrics.esg_score ?? "--");
  setText("supplier-compliance", profile.metrics.compliance ?? "--");
  setText("supplier-financial", profile.metrics.financial ?? "--");
  setText("supplier-resilience", profile.metrics.resilience ?? "--");
}

function renderDetailedRiskTrend(trend) {
  const node = document.getElementById("detailed-risk-trend");
  if (!node || !trend) return;

  const traces = [];
  if (trend.actual?.x?.length) {
    traces.push({
      x: trend.actual.x,
      y: trend.actual.y,
      mode: "lines+markers",
      name: "Actual",
      line: { color: "#1f77b4", width: 3 },
    });
  }
  if (trend.forecast?.x?.length) {
    const startValue =
      trend.actual?.y?.[trend.actual.y.length - 1] ?? trend.forecast.y[0];
    const forecastX = [trend.actual?.x?.slice(-1)[0], ...trend.forecast.x];
    const forecastY = [startValue, ...trend.forecast.y];
    traces.push({
      x: forecastX,
      y: forecastY,
      mode: "lines",
      name: "Forecast",
      line: { color: "#1f77b4", width: 3, dash: "dot" },
    });
  }

  const layout = {
    margin: { t: 10, r: 10, b: 30, l: 40 },
    xaxis: { gridcolor: "#edf1fa" },
    yaxis: { range: [0, 100], gridcolor: "#edf1fa" },
    showlegend: true,
  };

  Plotly.react(node, traces, layout, { displayModeBar: false, responsive: true });
}

function renderEsgBreakdown(data) {
  const node = document.getElementById("esg-breakdown-chart");
  if (!node || !data) return;

  const labels = ["Environmental", "Social", "Governance"];
  const values = [
    data.environmental ?? 0,
    data.social ?? 0,
    data.governance ?? 0,
  ];

  const trace = {
    x: labels,
    y: values,
    type: "bar",
    marker: { color: ["#2ca02c", "#ff7f0e", "#9467bd"] },
  };

  const layout = {
    margin: { t: 10, r: 10, b: 40, l: 40 },
    yaxis: { range: [0, 100], gridcolor: "#edf1fa" },
  };

  Plotly.react(node, [trace], layout, { displayModeBar: false, responsive: true });
}

function renderReliability(data) {
  if (!data) return;
  const setText = (id, value) => {
    const el = document.getElementById(id);
    if (el) el.textContent = value ?? "--";
  };
  setText("reliability-on-time", data.on_time);
  setText("reliability-defect", data.defect_rate);
  setText("reliability-lead-time", data.lead_time);
  setText("reliability-index", data.operational_index);
}

function renderRiskDrivers(drivers) {
  const node = document.getElementById("risk-drivers-chart");
  if (!node || !drivers || !drivers.length) return;

  const sorted = drivers.sort((a, b) => b.value - a.value);
  const data = [
    {
      x: sorted.map((d) => d.value),
      y: sorted.map((d) => d.label),
      type: "bar",
      orientation: "h",
      marker: { color: "#1f4e79" },
    },
  ];

  const layout = {
    margin: { t: 10, r: 10, b: 30, l: 80 },
    xaxis: { gridcolor: "#edf1fa" },
  };

  Plotly.react(node, data, layout, { displayModeBar: false, responsive: true });
}


function attachOverviewHandlers() {
  destroyMapListeners();

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