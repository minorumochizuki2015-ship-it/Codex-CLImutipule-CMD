# MVP 実装計画（並列CLI + API専用モード）

## スコープ
- ローカルLLMを使用しないAPI専用モード（`LLM_ENABLED=false`）。
- Codex CLI / Cline / Cursor / Gemini CLI の並列運用。
- MCP HTTPサーバーと各CLI統合の安定化（ポート競合解決、最小キャッシュ設定）。

## 作業タスク
- T1: `scripts/start_parallel_cli.ps1` にAPI専用環境変数適用とポート競合回避（近傍探索）を確認。
- T2: `.env.example` のLLM関連項目（`LLM_ENABLED=false`, `LLM_CACHE_BACKEND=memory`, `LLM_CACHE_REDIS_URL=`）を明示。
- T3: `src/mcp_agent_mail/llm.py` に `settings.llm.enabled==false` で短絡復帰ガード（APIキー未設定時の安全化）を確認。
- T4: CLI統合スクリプトの実行確認（`integrate_codex_cli.sh`, `integrate_cline.sh`, `integrate_cursor.sh`, `integrate_gemini_cli.sh`）。
- T5: `docker-compose.yml` の `8765:8765` 公開確認とREADME反映。
- T6: 運用ガイドの更新（マルチコンソール手順、トークン適用、ログ方針）。

## 検証
- V1: MCP HTTPサーバー起動ログ確認（`uvicorn` 稼働と設定反映）。
- V2: CLI起動コンソールがそれぞれ展開されることを確認。
- V3: APIキー未設定時にLLM補完が呼ばれないこと（短絡復帰）。
- V4: 8765競合時に代替ポートが自動選択されること。

## 受け入れ基準
- MCPサーバーと4種CLIが並列起動可能。
- APIキーなしでも安全に運用可能（LLM無効ガード適用）。
- ポート競合時も自動回避し、ドキュメントに手順が反映。

## 留意事項
- 機密情報をログや文書へ出さない（`.env`管理）。
- 行末はLF推奨（`.gitattributes` で `* text=auto eol=lf` を設定）.