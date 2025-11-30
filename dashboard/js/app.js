let currentRegion = null;
let currentTab = "overview";
let mapInitialized = false;

// DOM element references
let mapContainer;
let resetButton;
let regionFilter;

const tabButtons = document.querySelectorAll(".tab-button");
const tabContent = document.getElementById("tab-content");

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
    initOverviewTab();
  } else if (tabId === "detailed") {
    // Future: initialize detailed view
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

async function fetchTrendData() {
  const response = await fetch("/api/trend-data");
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

async function renderTrendChart() {
  const node = document.getElementById("trend-chart");
  if (!node) return;

  const trendData = await fetchTrendData();

  const data = trendData.series.map((s) => ({
    x: s.x,
    y: s.y,
    mode: "lines",
    name: s.name,
    line: {
      color: s.color,
      width: 3,
    },
  }));

  const layout = {
    margin: { t: 20, r: 10, b: 30, l: 40 },
    legend: {
      orientation: "h",
      y: -0.2,
    },
    xaxis: {
      title: "",
      gridcolor: "#edf1fa",
    },
    yaxis: {
      title: "",
      gridcolor: "#edf1fa",
    },
  };

  const config = {
    displayModeBar: false,
    responsive: true,
  };

  Plotly.react(node, data, layout, config);
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