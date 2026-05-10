import sqlite3
import os

DB_PATH = os.getenv("RESEARCH_DB_PATH", "./data/research.db")


def seed():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS papers (
            id INTEGER PRIMARY KEY,
            title TEXT,
            authors TEXT,
            year INTEGER,
            topic TEXT,
            citations INTEGER,
            abstract TEXT
        )
    """)

    papers = [
        ("Climate Tipping Points and Ocean Systems", "Hansen et al.", 2021, "climate change", 342, "Examines feedback loops in ocean systems triggered by rising CO2 levels."),
        ("Deep Learning for Climate Prediction", "Zhang, Liu", 2022, "machine learning", 210, "Applies transformer models to long-range climate forecasting with high accuracy."),
        ("Ocean Acidification Effects on Coral Reefs", "Morrison, Patel", 2020, "climate change", 517, "Studies the impact of pH reduction on coral bleaching events globally."),
        ("GPT Models in Scientific Research", "OpenAI Research", 2023, "machine learning", 890, "Evaluates large language models as research assistants across scientific domains."),
        ("Arctic Ice Melt Acceleration", "Petrov, Chen", 2022, "climate change", 430, "Documents record ice loss rates in the Arctic from 2015 to 2022."),
        ("Reinforcement Learning for Energy Grids", "Kim, Osei", 2021, "machine learning", 175, "Uses RL agents to optimize renewable energy distribution in smart grids."),
        ("Sea Level Rise Projections 2100", "IPCC Team", 2023, "climate change", 1200, "Updated projections show 0.5-1.0m rise by 2100 under current emission scenarios."),
        ("Multimodal LLMs for Data Analysis", "Rivera, Ahmed", 2023, "machine learning", 305, "Combines vision and language models for automated scientific data interpretation."),
        ("Permafrost Thaw and Methane Release", "Ivanova, Brooks", 2022, "climate change", 388, "Quantifies methane emissions from thawing permafrost in Siberia and Alaska."),
        ("Few-Shot Learning in Medical Diagnosis", "Nguyen, Park", 2021, "machine learning", 260, "Demonstrates few-shot learning for rare disease classification with limited data."),
    ]

    cursor.executemany(
        "INSERT OR IGNORE INTO papers (title, authors, year, topic, citations, abstract) VALUES (?,?,?,?,?,?)",
        papers,
    )

    conn.commit()
    conn.close()
    print(f"[seed] Research database seeded at {DB_PATH}")


if __name__ == "__main__":
    seed()