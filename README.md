# Brent Oil Price Analysis and Change Point Detection
## Project Overview
This project investigates the impact of significant political, economic, and regulatory events on Brent oil prices. Using change point analysis and time series modeling, we aim to provide data-driven insights that help investors, policymakers, and energy companies understand market fluctuations and make informed decisions. This project is structured to provide actionable intelligence and improve forecasting accuracy in the highly volatile oil market.

## Business Context
Birhan Energies, a consultancy specializing in energy sector insights, requires an analysis of Brent oil prices in relation to key events over the past decade. The findings will support decision-making for various stakeholders by exploring the effects of geopolitical and economic factors on oil price trends.

## Key Objectives
Identify major events that have affected Brent oil prices.
Quantify the impact of these events on price changes.
Deliver insights to guide investment, policy development, and operational planning.
### Data
The dataset includes historical Brent oil prices from May 20, 1987, to September 30, 2022, with daily records of price per barrel in USD.

#### Data Fields
- Date: The date of the recorded price.
- Price: The daily price of Brent oil in USD.
## Project Structure
The analysis is divided into the following tasks:

#### Task 1: Data Analysis Workflow and Model Understanding
Define the Workflow: Outline steps and processes, understand model inputs and outputs, and state assumptions.
Understand Models: Familiarize with time series models (e.g., ARIMA, GARCH) for analyzing price fluctuations.
#### Task 2: Analyzing Brent Oil Prices and Exploring Influential Factors

1. Time Series Analysis: Build on foundational knowledge to analyze historical data.
2. Advanced Models:
- **VAR (Vector Autoregression)**: For multivariate analysis.
- **Markov-Switching ARIMA:** For detecting different market conditions.
- **LSTM (Long Short-Term Memory):** For capturing complex patterns.
3. Explore Additional Factors:
- Economic indicators (GDP, inflation, exchange rates).
- Technological changes (advancements in extraction, renewable energy).
- Political factors (trade policies, environmental regulations).

#### Task 3: Dashboard Development
Develop an interactive dashboard using Flask (backend) and React (frontend) for stakeholders to explore and visualize the effects of various events on oil prices.

* Backend: Develop APIs for data access, manage data requests.
* Frontend: Design interactive visualizations (e.g., filters, date ranges, event highlights).

## Technologies and Tools
`Python`: For data analysis and modeling (e.g., using PyMC3 for Bayesian inference).

`PyMC3`: Bayesian modeling and change point detection.

`Flask`: Backend API for the dashboard.

`React`: Frontend for an interactive user interface.

`Data Visualization:` Recharts, React Chart.js 2, D3.js.