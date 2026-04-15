/**
 * AI Stock Intelligence Dashboard
 * JavaScript logic for API integration and data visualization
 */

let predictionChart = null;

// DOM Elements
const tickerInput = document.getElementById('tickerInput');
const predictBtn = document.getElementById('predictBtn');
const dashboard = document.getElementById('dashboard');
const loader = document.getElementById('loader');
const errorMessage = document.getElementById('errorMessage');
const suggestionsContainer = document.getElementById('suggestions');

// Predefined Stock List (STATIC)
const stocks = [
  { name: "Apple", symbol: "AAPL" },
  { name: "Microsoft", symbol: "MSFT" },
  { name: "Google", symbol: "GOOG" },
  { name: "Amazon", symbol: "AMZN" },
  { name: "Tesla", symbol: "TSLA" },
  { name: "NVIDIA", symbol: "NVDA" },
  { name: "Reliance", symbol: "RELIANCE.NS" },
  { name: "TCS", symbol: "TCS.NS" },
  { name: "Infosys", symbol: "INFY.NS" }
];

// API Base URL
const API_BASE = "";

/**
 * Main function to fetch data and update the UI
 */
async function analyzeAsset() {
    const ticker = tickerInput.value.trim().toUpperCase();
    if (!ticker) {
        showError("Please enter a valid ticker symbol.");
        return;
    }

    const heroAnimation = document.getElementById('heroAnimation');
    if(heroAnimation) heroAnimation.style.display = 'none';

    // Disable button to prevent multi-submissions
    predictBtn.disabled = true;
    predictBtn.style.opacity = '0.5';
    predictBtn.textContent = 'Processing...';

    // Reset UI state
    hideError();
    showLoader();
    dashboard.style.display = 'none';

    try {
        const response = await fetch(`${API_BASE}/predict?ticker=${ticker}`);
        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail?.error || "Failed to fetch prediction data.");
        }

        updateDashboard(data);
        hideLoader();
        dashboard.style.display = 'grid';
        
        
    } catch (err) {
        console.error(err);
        showError(err.message);
        hideLoader();
    } finally {
        // Re-enable button
        predictBtn.disabled = false;
        predictBtn.style.opacity = '1';
        predictBtn.textContent = 'Analyze Assets';
    }
}

/**
 * Updates all dashboard cards with the retrieved API data
 */
function updateDashboard(data) {
    // 1. Update Price Labels
    document.getElementById('currentPrice').textContent = `$${data.current_price.toLocaleString()}`;
    document.getElementById('avgPredicted').textContent = `$${data.average_predicted_price.toLocaleString()}`;

    // 2. Render Chart
    renderChart(data.predictions);

    // 3. Update Signal Card
    const signalDisplay = document.getElementById('signalDisplay');
    signalDisplay.textContent = data.final_decision;
    signalDisplay.className = `signal-value signal-${data.final_decision}`;

    // 4. Update Confidence Card
    const confPercent = Math.round(data.confidence * 100);
    document.getElementById('confidenceValue').textContent = `${confPercent}%`;
    document.getElementById('confidenceBar').style.width = `${confPercent}%`;

    // 5. Update Risk Card
    const riskDisplay = document.getElementById('riskLevel');
    riskDisplay.textContent = data.risk_level;
    riskDisplay.className = `risk-level risk-${data.risk_level}`;

    // 6. Update Reason Text
    document.getElementById('reasonText').textContent = data.reason;

    // 7. Update Model Accuracy Note
    const accuracyNote = document.getElementById('accuracyNote');
    if (accuracyNote && data.model_accuracy_note) {
        accuracyNote.textContent = data.model_accuracy_note;
        accuracyNote.style.display = 'block';
    }
}

/**
 * Initializes or updates the Chart.js instance with forecast data
 */
function renderChart(predictions) {
    const ctx = document.getElementById('predictionChart').getContext('2d');
    
    const labels = predictions.map(p => p.date);
    const prices = predictions.map(p => p.predicted_price);

    // If chart already exists, destroy it before creating a new one
    if (predictionChart) {
        predictionChart.destroy();
    }

    // Custom Chart.js Theme for Dark Mode
    predictionChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: 'Predicted Closing Price ($)',
                data: prices,
                borderColor: '#3b82f6',
                backgroundColor: 'rgba(59, 130, 246, 0.1)',
                borderWidth: 3,
                pointBackgroundColor: '#3b82f6',
                pointBorderColor: '#fff',
                pointHoverRadius: 6,
                tension: 0.3,
                fill: true
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: '#1a1b23',
                    titleFont: { size: 14, weight: 'bold' },
                    bodyFont: { size: 13 },
                    padding: 12,
                    displayColors: false,
                    borderColor: 'rgba(255,255,255,0.1)',
                    borderWidth: 1
                }
            },
            scales: {
                x: {
                    grid: { color: 'rgba(255,255,255,0.05)' },
                    ticks: { color: '#a0a0b0', font: { family: 'Outfit' } }
                },
                y: {
                    grid: { color: 'rgba(255,255,255,0.05)' },
                    ticks: { color: '#a0a0b0', font: { family: 'Outfit' } }
                }
            }
        }
    });
}

// UI Utility Functions
function showLoader() { 
    loader.style.display = 'block'; 
    document.body.classList.add('searching');
}
function hideLoader() { 
    loader.style.display = 'none'; 
    document.body.classList.remove('searching');
}
function showError(msg) {
    errorMessage.textContent = msg;
    errorMessage.style.display = 'block';
}
function hideError() { errorMessage.style.display = 'none'; }

// Event Listeners
predictBtn.addEventListener('click', analyzeAsset);
tickerInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
        analyzeAsset();
        hideSuggestions();
    }
});

// Autocomplete Logic
tickerInput.addEventListener('input', () => {
    const query = tickerInput.value.trim().toLowerCase();
    if (!query) {
        hideSuggestions();
        return;
    }

    const filtered = stocks.filter(s => 
        s.symbol.toLowerCase().includes(query) || 
        s.name.toLowerCase().includes(query)
    ).slice(0, 5);

    if (filtered.length > 0) {
        renderSuggestions(filtered);
    } else {
        hideSuggestions();
    }
});

function renderSuggestions(list) {
    suggestionsContainer.innerHTML = '';
    list.forEach(stock => {
        const item = document.createElement('div');
        item.className = 'suggestion-item';
        item.innerHTML = `
            <span class="stock-symbol">${stock.symbol}</span>
            <span class="stock-name">${stock.name}</span>
        `;
        item.addEventListener('click', () => {
            tickerInput.value = stock.symbol;
            hideSuggestions();
        });
        suggestionsContainer.appendChild(item);
    });
    suggestionsContainer.style.display = 'block';
}

function hideSuggestions() {
    suggestionsContainer.style.display = 'none';
}

// Hide suggestions when clicking outside
document.addEventListener('click', (e) => {
    if (!tickerInput.contains(e.target) && !suggestionsContainer.contains(e.target)) {
        hideSuggestions();
    }
});

// Optional: Initial Load with Default Ticker (e.g. BTC-USD or AAPL)
// window.onload = () => { tickerInput.value = 'AAPL'; analyzeAsset(); };
