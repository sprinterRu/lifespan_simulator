#!/usr/bin/env bash

exec uv run streamlit run app.py --server.fileWatcherType poll --server.runOnSave true "$@"
