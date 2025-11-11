# MVP 詳細技術設計（API専用モード＋並列CLI）

## 構成要素
- MCP HTTP Server: `uvicorn mcp_agent_mail.http:build_http_app --factory`（既定 `8765`）。
- 起動スクリプト: `scripts/start_parallel_cli.ps1`（Codex/Cline/Cursor/Gemini を個別コンソールで起動）。
- LLM設定: `src/mcp_agent_mail/config.py`（`LlmSettings`）。
- LLM呼び出し: `src/mcp_agent_mail/llm.py`（LiteLLM統合、コストログ、キャッシュ）。

## API専用モード仕様
- 強制環境: `LLM_ENABLED=false`, `LLM_CACHE_BACKEND=memory`, `LLM_CACHE_REDIS_URL=""`。
- ガード動作: `settings.llm.enabled==false` の場合、LLM補完関数は早期復帰（空またはNo-Op）。
- キャッシュ: 既定はメモリ、`LLM_CACHE_REDIS_URL` 設定時はRedis使用可能（現運用では空）。
- コスト記録: `success_callback` で呼び出しコストを記録（API専用時はキー設定時のみ発火）。

## ポート競合回避
- 既定 `8765` 使用。競合検出時は近傍（+1〜+20）で空きポートを自動検索。
- 検出: Windowsの `Get-NetTCPConnection` を利用。
- 起動: `uv run python -c "import uvicorn; uvicorn.run('mcp_agent_mail.http:build_http_app', factory=True, host='0.0.0.0', port=$ServerPort)"`。

## CLI統合詳細
- Codex: `scripts/integrate_codex_cli.sh --yes`（MCPエンドポイント検出→トークン生成→登録）。
- Cline: `scripts/integrate_cline.sh --yes`。
- Cursor: `scripts/integrate_cursor.sh --yes`。
- Gemini: `scripts/integrate_gemini_cli.sh --yes`。
- すべて、`scripts/run_server_with_token.sh` の再利用で起動を簡略化。

## 運用フロー
1. PowerShellで `start_parallel_cli.ps1` を起動（必要なCLIフラグ指定）。
2. 最初のコンソールでMCPサーバーの稼働ログ確認。
3. 各CLIのMCP統合を確認し、必要に応じて外部APIキーを設定。
4. ローカルLLMは無効化されたまま、外部APIのみで推論実行。

## セキュリティ/運用
- 機密情報は `.env`/環境変数で管理。ログ/文書出力禁止。
- `.gitattributes` にて `* text=auto eol=lf` を推奨、LF統一。
- Windows前提のため、ポート監視は `Get-NetTCPConnection` を利用する。