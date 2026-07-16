---
title: "エージェント設計・オーケストレーションをハンズオンで学ぶ"
emoji: "🤖"
type: "tech"
topics: ["claude", "anthropic", "ai", "python", "agent"]
published: true
---

## はじめに

Anthropicが2026年3月に開始した技術認定資格「Claude Certified Architect(CCA-F)」では、**エージェント設計・オーケストレーション**が最も比重の高い出題領域とされています。座学だけで理解するのは難しい分野なので、実際に手を動かしながら基礎パターンを体に染み込ませることにしました。

この記事では、環境構築からつまずいたポイント、4段階のハンズオン(単発ツール呼び出し→エージェント・ループ→並列ツール実行→マルチエージェント・オーケストレーション)までを、実際に踏んだ手順そのままに解説します。

対象読者:
- CCA-F受験を検討している方
- Claude APIでエージェントを構築し始めたい方
- Claude Codeをこれから使い始める方

本記事で使用したコードは以下のリポジトリで公開しています。
https://github.com/qameqame/agentic-orchestration-tutorial

## 環境構築でつまずいたこと

### Claude Codeのインストール

Claude Codeは以下のコマンドでインストールできます。

```bash
npm install -g @anthropic-ai/claude-code
```

補足：nodenvを使用している方は、グローバルインストールしたパッケージのシム(shim)を手動で再生成する必要があります。

```bash
nodenv rehash
claude --version
# => 2.1.211 (Claude Code)
```

### ログイン方法の選び方

Claude Codeには2つの認証方法があります。

| 方法 | 対象 | 課金 |
|---|---|---|
| Claude.aiログイン(OAuth) | Pro/Max/Team/Enterprise契約者 | サブスクリプション内 |
| APIキー(Console) | 従量課金を使いたい開発者 | pay-per-token |

**重要な注意点**: システムに `ANTHROPIC_API_KEY` という環境変数が設定されていると、Pro/Max契約中でもそちらが優先され、意図せずAPI課金が発生します。サブスクリプションの範囲で使いたいなら、この環境変数は設定しないのが正解です。

Claude Pro契約であれば、ターミナルで `claude` を実行し、「Claude.ai」でログインするだけで完了します。

```bash
claude
# ブラウザが開くので、Claude Proアカウントでサインイン
```

### 落とし穴: Claude Pro ≠ 素のMessages API

ここが最も勘違いしやすいポイントでした。**Claude Code(CLI)はProサブスクリプションでカバーされますが、`anthropic` Python SDKで直接 `api.anthropic.com` を叩く「素のMessages API」は別課金**です。

つまり、これから紹介する自作Pythonスクリプト(Ring1〜4)を動かすには、Claude Proとは別に、Anthropic ConsoleでAPIキーを発行し、少額のクレジットをチャージする必要があります。

```
Claude Code(CLI)         → Proサブスクリプションでカバー
自作Pythonスクリプト(SDK) → Console APIキー + 従量課金が別途必要
```

「Proに入っているのに、なぜまたお金がかかるのか」と最初は混乱しましたが、CLIと素のAPIは別の課金体系だと理解してからは腹落ちしました。

## ハンズオン: 4つのRingで学ぶエージェントパターン

ここからは、`anthropic` Python SDKを使って、単純なツール呼び出しから本格的なマルチエージェント構成まで、1つずつ概念を積み上げていきます。

### Ring 1: 単発のツール呼び出し

最小構成として、ツール1つ・メッセージ1つ・呼び出し1つを試します。

```python
import anthropic

client = anthropic.Anthropic()

tools = [{
    "name": "get_weather",
    "description": "指定した都市の現在の天気を取得する",
    "input_schema": {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    },
}]

response = client.messages.create(
    model="claude-sonnet-5",
    max_tokens=1024,
    tools=tools,
    messages=[{"role": "user", "content": "東京の天気を教えて"}],
)

print("stop_reason:", response.stop_reason)
for block in response.content:
    if block.type == "tool_use":
        print("呼び出されたツール:", block.name)
        print("引数:", block.input)
```

実行結果:

```
stop_reason: tool_use
呼び出されたツール: get_weather
引数: {'city': 'Tokyo'}
```

ここで理解すべき一番重要な点は、**Claude自身はツールを実行しない**ということです。`stop_reason: "tool_use"` は「このツールをこの引数で呼びたい」という意思表示に過ぎず、実行するかどうかはアプリケーション側の責任です。

### Ring 2: エージェント・ループ

現実のタスクは1回のツール呼び出しでは完結しません。結果を見て次の行動を決める必要があるため、`while` ループと会話履歴の蓄積が必要になります。

```python
messages = [{"role": "user", "content": "東京と大阪、どっちが暖かい?"}]

while True:
    response = client.messages.create(
        model="claude-sonnet-5",
        max_tokens=1024,
        tools=tools,
        messages=messages,
    )
    messages.append({"role": "assistant", "content": response.content})

    if response.stop_reason != "tool_use":
        final_text = "".join(b.text for b in response.content if b.type == "text")
        print("最終回答:", final_text)
        break

    tool_results = []
    for block in response.content:
        if block.type == "tool_use":
            result = run_tool(block.name, block.input)
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": result,
            })
    messages.append({"role": "user", "content": tool_results})
```

#### つまずきポイント: ダミーデータのキー不一致

最初の実行では、こんな結果になりました。

```
最終回答: 申し訳ありません。東京・大阪ともに天気データを取得できませんでした
```

原因は単純で、ダミーデータの辞書キーが英語(`"Tokyo"`)だったのに対し、Claudeが日本語の質問文からそのまま `city="東京"` として引数を組み立てていたため、辞書に該当キーがなく `get()` がデフォルト値(データなし)を返していたのです。

```python
# Before(英語キーのみ)
fake_data = {"Tokyo": "晴れ、28度", "Osaka": "曇り、25度"}

# After(日英両対応)
fake_data = {
    "Tokyo": "晴れ、28度", "東京": "晴れ、28度",
    "Osaka": "曇り、25度", "大阪": "曇り、25度",
}
```

修正後は無事に動作しました。

```
最終回答: 現在の天気を比較すると:
- 東京: 晴れ、28度
- 大阪: 曇り、25度
東京の方が暖かいです(3度差)。
```

この一件は、CCA-Fの「ツール設計」領域でも問われる重要な教訓です。ツールのスキーマに入力形式(例: 「都市名は英語で」)を明記する、またはツール側で入力を正規化する設計が必要になります。実運用では想定外の入力パターンをどう吸収するかが、エージェントの信頼性を左右します。

### Ring 3: 複数ツールの並列実行

Claudeは1ターンで複数のツールを同時に呼び出すことがあります。これを効率的に処理するには `ThreadPoolExecutor` などで並列実行します。

```python
import concurrent.futures

def handle_tool_calls(response):
    tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(run_tool, b.name, b.input): b
            for b in tool_use_blocks
        }
        return [
            {
                "type": "tool_result",
                "tool_use_id": futures[f].id,
                "content": f.result(),
            }
            for f in futures
        ]
```

天気に加えて人口も取得するツールを追加し、「東京と大阪の天気と人口を教えて」と質問すると、1ターンで4つのツール呼び出し(天気×2、人口×2)が発生し、並列実行されます。

```
## 東京
- 天気: 晴れ、28度
- 人口: 約1,400万人
## 大阪
- 天気: 曇り、25度
- 人口: 約900万人
```

**設計上の注意点**として、読み取り専用のツール(検索・参照)は並列実行に向きますが、状態を変更するツール(書き込み・送信・削除)は実行順序が結果に影響するため、直列実行にすべきという原則があります。

### Ring 4: マルチエージェント・オーケストレーション

最後は、1つのClaudeがすべてを担うのではなく、「オーケストレーター」が「ワーカー」にタスクを委任するパターンです。

```python
def worker_agent(task: str) -> str:
    """独立したコンテキストで単一のサブタスクを実行するワーカー"""
    response = client.messages.create(
        model="claude-sonnet-5",
        max_tokens=1024,
        messages=[{"role": "user", "content": task}],
    )
    return "".join(b.text for b in response.content if b.type == "text")

def orchestrator(goal: str) -> str:
    # ステップ1: タスクをサブタスクに分解
    planning = client.messages.create(
        model="claude-sonnet-5",
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": f"次の目標を独立して実行可能な2〜3個のサブタスクに分解して: {goal}"
        }],
    )
    plan_text = "".join(b.text for b in planning.content if b.type == "text")
    subtasks = [line.strip("- ").strip() for line in plan_text.splitlines() if line.strip()]

    # ステップ2: 各サブタスクを独立したワーカーに委任
    results = [worker_agent(task) for task in subtasks]

    # ステップ3: 結果を統合
    synthesis = client.messages.create(
        model="claude-sonnet-5",
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": f"以下の結果を統合して、目標『{goal}』への最終回答をまとめて:\n\n" + "\n\n".join(results)
        }],
    )
    return "".join(b.text for b in synthesis.content if b.type == "text")
```

「新商品のローンチプランを作成する」という目標を投げると、オーケストレーターが「市場・競合分析」「プロモーション設計」「チャネル・在庫・価格戦略」のようなサブタスクに分解し、それぞれ独立したワーカーが検討した内容を、最後に1つのプランへ統合してくれました。

**なぜ独立したワーカーを使うのか**。同じセッション内でClaudeに自己レビューさせると、直前の推論を引きずってしまい、誤りを見逃しやすくなります。前の文脈を持たない別インスタンスに検証や実行を任せる方が、微妙な問題を発見しやすいというのが、この設計の狙いです。

## 4つのRingを終えて

| Ring | 概念 | つまずき |
|---|---|---|
| 1 | 単発ツール呼び出し、`stop_reason`の意味 | なし |
| 2 | `while`ループ、会話履歴の蓄積 | ダミーデータの日英キー不一致 |
| 3 | 1ターン内の複数ツール呼び出しと並列実行 | 同上(要修正) |
| 4 | オーケストレーター/ワーカー構成 | なし |

振り返ると、一番学びになったのは「動かなかった瞬間」でした。Ring2で天気データが取得できなかった原因を追うことで、ツールのスキーマ設計と入力の正規化が実運用でどれだけ重要かを実感できました。教科書的な説明だけでは得られない気づきです。

## 次にやりたいこと

- **Tool Runner(SDK組み込み)への置き換え**: 今回は手書きの`while`ループでしたが、SDKに用意されたTool Runnerを使うとコード量が半分程度になります。同じ概念がどう簡潔になるか比較してみたい。
- **Claude Agent SDKへの移行**: 2026年6月15日から、Pro/Max/Team/Enterprise契約者はAgent SDK用の月間クレジットを無料で受け取れるようになりました。素のMessages APIとは別課金だった今回のスクリプトを、サブスクリプション内で完結する形に書き換えられないか試したいです。
- **MCP(Model Context Protocol)連携**: 外部ツール・データソースへの接続を標準化する仕組みで、CCA-Fの「ツール設計・MCP統合」領域にも直結します。

## 参考ドキュメント

- サンプルコード一式(本記事のリポジトリ): https://github.com/qameqame/agentic-orchestration-tutorial
- Tool use: https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview
- Agent SDKのループ解説: https://platform.claude.com/docs/en/agent-sdk/agent-loop
- Claude Codeのループ設計: https://code.claude.com/docs/en/how-claude-code-works

## おわりに

エージェント設計・オーケストレーションは、概念自体はシンプルですが、実際に手を動かすと「ツールの入力設計」「並列実行と直列実行の使い分け」「コンテキスト分離の効果」など、実務で効いてくる細かい判断が随所に出てきます。CCA-F対策として始めましたが、業務でエージェントを構築する上でもそのまま使える知識になったと感じています。
