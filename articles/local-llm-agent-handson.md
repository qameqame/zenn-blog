---
title: "ローカルLLM(Ollama + Qwen2.5)でAIエージェントを自作するハンズオン"
emoji: "🤖"
type: "tech"
topics: ["ollama", "llm", "ai", "agent", "docker", "python"]
published: true
---

## はじめに

クラウドAPIのコストや情報漏洩リスクを気にせず、AIエージェントの仕組みを手を動かして理解したい——という動機で、ローカルLLM(Ollama + Qwen2.5)だけでAIエージェントを組み立てる1〜2時間のハンズオンを作りました。

フレームワーク(LangGraph、CrewAIなど)は便利ですが、最初にブラックボックスのまま使ってしまうと「エージェントが何をしているのか」が掴みにくくなります。このハンズオンでは、あえてフレームワークを使わず、`LLM + ツール呼び出しのforループ`という最小構成を自分の手で書くことで、フレームワークが裏で何を肩代わりしているのかを体感できるようにしています。

コードとハンズオン資料一式はこちらです(READMEは英語ですが、日本語版は`README.ja.md`)。

https://github.com/qameqame/local-llm-agent-handson

## 対象読者

- Python経験があるエンジニア
- クラウドAPIを使わずローカルだけでAIエージェントを試したい人
- 「エージェントって結局中で何をやっているの?」を実装レベルで理解したい人

## 環境

- Ollama(ネイティブ or Docker)
- モデル: `qwen2.5:14b-instruct-q4_K_M`(tool calling対応)
- Python 3.10+ / `ollama` Pythonライブラリ

Docker版Ollamaでも、ネイティブインストールでも同じ手順で進められます。筆者はDocker(`ollama/ollama`イメージ)上で動かしています。

```bash
docker ps --filter "name=ollama"
docker exec -it <コンテナ名> ollama list
```

## 全体構成

所要時間1〜2時間を想定し、5ステップに分けています。

| ステップ | 内容 |
|---|---|
| 0 | セットアップ・疎通確認 |
| 1 | 単発のツール呼び出し |
| 2 | マルチターン・エージェントループ |
| 3 | 実践: ローカルファイル調査エージェントを自作 |
| 4 | 安全装置(エラーハンドリング・打ち切り条件) |
| 5 | まとめ・次のステップ |

以下、それぞれのポイントをかいつまんで紹介します。

## ステップ1: 単発のツール呼び出し

Ollamaのtool calling APIの良いところは、Pythonの関数をそのまま`tools`リストに渡せることです。型ヒントとdocstringがそのままツールのスキーマになります。

```python
def get_temperature(city: str) -> str:
    """指定した都市の現在の気温を取得する

    Args:
        city: 都市名(英語表記)

    Returns:
        現在の気温
    """
    temperatures = {"Tokyo": "31°C", "New York": "22°C", "London": "15°C"}
    return temperatures.get(city, "不明な都市です")

response = chat(model=MODEL, messages=messages, tools=[get_temperature])
```

モデルが「ツールを呼びたい」と判断すると`response.message.tool_calls`に呼び出し内容が入って返ってきます。こちらで実際に関数を実行し、結果を`role: "tool"`のメッセージとして会話履歴に追加し、もう一度`chat()`を呼ぶと最終回答が返る——という流れです。

「LLMがツールを選ぶ→アプリ側が実行する→結果をLLMに戻す」というAIエージェントの最小単位を、ここで生のコードとして体験できます。

## ステップ2: マルチターン・エージェントループ

実際のタスクは1回のツール呼び出しでは終わりません。そこで`while True`ループにして、モデルがツール呼び出しをやめるまで繰り返します。

```python
while True:
    response = chat(model=MODEL, messages=messages, tools=[add, multiply])
    messages.append(response.message)
    if response.message.tool_calls:
        for tc in response.message.tool_calls:
            result = available_functions[tc.function.name](**tc.function.arguments)
            messages.append({"role": "tool", "tool_name": tc.function.name, "content": str(result)})
    else:
        break  # ツール呼び出しがなくなったら終了 = 最終回答
```

これがあらゆるエージェントフレームワークの内部にある最小構成です。ここが理解できると、フレームワークが自動でやってくれている処理(状態管理・リトライ・並列実行など)の正体が見えてきます。

### 実際に起きたハマりどころ: 引数のハルシネーション

このハンズオンを実際にqwen2.5:14b-instruct-q4_K_Mで動かしたところ、興味深い(そして教材として価値のある)失敗が起きました。

`(11434 + 12341) * 412`という計算を`add`→`multiply`の2ステップで解かせようとしたところ、モデルが`add`と`multiply`を同じターンで並列に呼び出し、`add`の結果がまだ分からない段階で`multiply`の引数に`"__SUM_OF_PREVIOUS_STEP__"`というプレースホルダー文字列を入れてしまったのです。

これをそのまま実行すると何が起きるか。Pythonは`"文字列" * 412`を「文字列を412回繰り返す」処理として実行してしまうため、**エラーにもならず**、意味不明な巨大文字列がツール結果として返ってしまいました。静かに壊れる、一番厄介なタイプのバグです。

対策として、`add`/`multiply`に引数の型検証を入れました。

```python
def _validate_numbers(a, b) -> None:
    for name, value in (("a", a), ("b", b)):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(
                f"引数 '{name}' には数値が必要ですが、{value!r} (型: {type(value).__name__}) が渡されました。"
                "前のツール呼び出しの結果を先に確認してから、実際の数値で再度呼び出してください。"
            )
```

さらに、ツール実行部分を`try/except`で囲み、検証エラーが起きてもクラッシュさせず、エラーメッセージをそのままモデルに返すようにしました。

```python
try:
    result = available_functions[tc.function.name](**tc.function.arguments)
except ValueError as e:
    result = f"エラー: {e}"
```

これにより、モデルは次のターンでエラー内容を見て、正しい数値(23775)を使って`multiply`を呼び直し、最終的に正しい答え(9,795,300)にたどり着けました。**「ツール呼び出しの引数は信用せず、必ず検証してから実行する」**——本番運用でも通用する、実装で得た教訓です。

## ステップ3: ローカルファイル調査エージェントを自作する

ここが今回のハンズオンのメインです。参加者自身に、以下の3つのツールを実装してもらいます。

- `list_files(directory)`: ディレクトリ内のファイル一覧を返す
- `read_file(path)`: ファイルの中身を返す(長すぎる場合は切り詰める)
- `search_files(directory, keyword)`: ファイル群からキーワードを含む行を検索する

これをエージェントループに渡すと、「`sample_docs/`フォルダの中にAPIキーの設定方法についてのメモはありますか?あれば要約して」という質問に対し、モデルが自律的に「一覧を見る→怪しいファイルを検索/特定→中身を読む→要約する」という多段階の行動をとれるようになります。

```
==== ターン1 ====
[ツール実行] list_files({'directory': './sample_docs'})
==== ターン2 ====
[ツール実行] read_file({'path': './sample_docs/setup_memo.txt'})
==== ターン3 ====
--- 最終回答 ---
`setup_memo.txt` には確かに API キーの設定方法について記載があります。...
```

3ターンで目的の情報にたどり着き、内容も原文と一致した要約が返ってきました。フレームワークなしでも、ツール設計とループさえきちんと書けば、実用的な多段階エージェントが作れることが確認できます。

### もう一つの実例: パスのハルシネーション

質問文を曖昧にしたまま実行すると、モデルが実在しないパス(`/filepath/sample_docs`のような架空の絶対パス)を推測してツールに渡すことがありました。ただし、`list_files`/`search_files`側でディレクトリの存在チェックをしていたため、エージェントはクラッシュせず、嘘の要約をでっちあげることもなく、「パスが違うようです、教えてください」と正直に聞き返す挙動になりました。

質問文に「`sample_docs`(カレントディレクトリ直下の相対パス)」と明示するだけで、この推測ミスはかなり減ります。ツールの堅牢性(存在チェック)と、プロンプトの具体性、その両方が効いてくる好例でした。

## ステップ4: 安全装置

実務で使うことを見据え、以下の安全策を入れています。

- **最大ターン数の制限**(`MAX_TURNS`): 無限ループ・無限ツール呼び出しの防止
- **不明なツール名のハンドリング**: 存在しないツール名を呼ばれてもクラッシュさせない
- **出力の切り詰め**: `read_file`はコンテキスト長を圧迫しないよう先頭数千文字に制限
- **引数の検証**: ステップ2で実演した通り、数値であるべき引数に文字列が来ることを想定する
- **破壊的操作の確認**: `write_file`のような操作をツール化する場合は、実行前にhuman-in-the-loopの確認を挟む

14Bクラスのモデルでも、ツール呼び出しの判断ミスや同じツールの無駄な繰り返しは普通に起きます。「エージェントは賢く振る舞う」ではなく「エージェントは時々おかしなことをする前提で、壊れても安全に倒れる設計にする」という視点が、ローカルLLMエージェントを実際に触ってみると強く実感できます。

## まとめ・次のステップ

今回書いたのは「LLM + ツール呼び出しのforループ」という、あらゆるエージェントフレームワークの内部にある最小構成です。ここが理解できていると、フレームワークが何を肩代わりしてくれているのかが見えるようになります。

次のステップとしては以下がおすすめです。

- **LangGraph**: 状態遷移・リトライ・チェックポイントを扱う本格的なエージェントを組みたい場合
- **CrewAI**: 役割(role)を持つ複数エージェントを協調させたい場合
- **モデルのスケールアップ**: より複雑な判断をさせたい場合、Qwen3 30B-A3B(MoE)やLlama 3.3 70Bなど大きいモデルの方が計画能力が高い
- **ハイブリッド構成**: 軽量なタスクはローカル、重要な判断だけクラウドのフロンティアモデルに投げる構成も実務では一般的

コードとハンズオン資料の全体は以下のリポジトリに置いています。手元のOllama環境があれば、そのまま1〜2時間で試せる構成になっています。

https://github.com/qameqame/local-llm-agent-handson

## 参考リンク

- [Ollama Tool calling 公式ドキュメント](https://docs.ollama.com/capabilities/tool-calling)
- [Ollama公式ブログ: Tool support](https://ollama.com/blog/tool-support)
