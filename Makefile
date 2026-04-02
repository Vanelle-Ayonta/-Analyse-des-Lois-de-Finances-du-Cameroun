# Makefile — Analyse Budget Cameroun
.PHONY: install run export lint format docker-build docker-run clean

# ── Installation ──────────────────────────────────────────────────────────────

install:
	@echo "📦 Installation des dépendances avec Poetry..."
	poetry install
	@echo ""
	@echo "✅ Prêt ! Lancez l'application avec :"
	@echo "   make run"

# ── Lancement ─────────────────────────────────────────────────────────────────

run:
	@echo "🚀 Lancement du tableau de bord..."
	poetry run streamlit run app.py

# ── Déploiement ───────────────────────────────────────────────────────────────

# Pour Streamlit Cloud ou serveurs sans Poetry
export:
	@echo "📋 Génération de requirements.txt..."
	poetry export -f requirements.txt --output requirements.txt --without-hashes
	@echo "✅ requirements.txt généré"

# ── Qualité du code ───────────────────────────────────────────────────────────

lint:
	poetry run ruff check app.py packs/

format:
	poetry run black app.py packs/

# ── Docker ────────────────────────────────────────────────────────────────────

docker-build:
	@echo "🐳 Build de l'image Docker..."
	docker build -t analyse-budget-cameroun .

docker-run:
	@echo "🐳 Lancement du container (port 8501)..."
	docker run -p 8501:8501 --env-file .env analyse-budget-cameroun

# ── Nettoyage ─────────────────────────────────────────────────────────────────

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -f tmp_uploaded_*.pdf
	@echo "🧹 Nettoyé"
