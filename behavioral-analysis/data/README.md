# Data

This directory contains the datasets used in the Behavioral Analysis and Brand Trust Scoring project.

## Dataset

The project uses the **e-commerce clickstream dataset (2019-Oct)**.

The raw dataset contains user interaction events collected from an e-commerce platform. These events are used to analyze user browsing and purchasing behavior and derive behavioral patterns for brands.

### Main Event Information

The dataset contains information such as:

- User ID
- Event timestamp
- Event type
- Product ID
- Category ID
- Category code
- Brand
- Product price
- User session ID

Typical event types include:

- `view`
- `cart`
- `purchase`

## Data Processing Pipeline

The raw clickstream data is processed through multiple stages:

```text
Raw Clickstream Data
        │
        ▼
Data Cleaning
        │
        ▼
Session Generation
        │
        ▼
Session-Level Features
        │
        ▼
Brand Behavior Features
        │
        ▼
Brand-Category Behavior
        │
        ▼
Category-Relative Deviations
        │
        ▼
Behavioral Evidence Filtering
        │
        ▼
Anomaly Detection
        │
        ▼
Brand Trust Score