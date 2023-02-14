# research_tabuchi
 
# 開発環境
Visual Studio Code(1.74.3)  
tweepy(4.10.0)  
Python(3.8.10)  
Google Colaboratory  
Twitter API academic  
# 前準備

1. Visual Studio Codeのダウンロードとインストール  
(https://www.javadrive.jp/vscode/install/index1.html)
2. Visual Studio CodeでPythonの環境を構築する  
(https://mtkbirdman.com/windows-visual-studio-code-python)
3. Twitter APIの取得  
(https://di-acc2.com/system/rpa/9688/)
4. Googleアカウントの準備  
(https://www.google.com/intl/ja/account/about/)

# 分析１の分類モデルの作成と評価
* `./analysis1/create_model`にある`over_cross_validation.ipynb`を実行　
* `./analysis1/create_model`にある`under_cross_validation.ipynb`を実行　
* `./analysis1/create_model`にオーバーサンプリング処理を加えたモデルとアンダーサンプリング処理を加えたモデルが作成される
* `./analysis1`にある`model_labeling.ipynb`を実行　
* `gakusyu_data.csv`にモデルによるラベル付けが行われる
　
* 参考サイト  
Simple Transformers 入門 (1) - テキスト分類：https://note.com/npaka/n/nfe2436ea5301  
機械学習における不均衡データへの対処方法（Over Sampling, Under Sampling）:https://book-read-yoshi.hatenablog.com/entry/2021/07/31/imbalanced_data_smote#%EF%BC%92%EF%BC%91Random-Under-Sampling 

# 分析1のトピックモデル
ここではVisual Studio Codeではなく、Google colaboratoryで実行する  
* `./analysis1/topic_model`直下にクラスターAからクラスターEまでのトピックモデルの処理コードがある
* それぞれ実行すると、LDAによるトピックモデルの結果が可視化で確認できる
* `./analysis1/topic_model/data`にはクラスターAからクラスターEのツイートデータ
* 参考サイト  
【自然言語処理】【Python】トピックモデル（LDA）を実装し、PyLDAvisを使ってインタラクティブに可視化する：https://zenn.dev/robes/articles/424cb97503d16e 
# ツイート収集
Twitter APIを取得しておく
* `./analysis2/user_identification`にある`collect_indicator_tweet.ipynb`に取得したTwitter APIを入力し、実行すると指標を含むツイートを収集する
```bash
def create_url(QUERY, MAX_RESULTS):
    # クエリ条件：収集する情報を指定、期間指定、単語の指定
    query = QUERY
    tweet_fields = "tweet.fields=author_id,text,created_at"
    start_time = 'start_time=2022-07-24T00:00:00Z'
    end_time = 'end_time=2022-10-24T00:00:00Z'
    max_results = MAX_RESULTS
    url = "https://api.twitter.com/2/tweets/search/all?query={}&{}&{}&{}&{}".format(
        query, tweet_fields,start_time,end_time,max_results
    )
    return url
```
* 上記のクエリに収集期間やキーワードを設定することでツイートを収集する  

* 参考サイト  
Developer Platform：https://developer.twitter.com/en  
Twitter API v2の使い方4[ユーザー情報取得編]:https://circleken.net/2021/06/post60/  

# 分析２のツイート分析
## TF-IDF
* `./analysis2/all_analysis_data/all_analysis_labeled_data/ketuatu_analysis/tfidf`にある`tf_idf.ipynb`を実行  
* 血圧が正常・高い・低い(1週間分)と血圧が正常・高い・低い(キーワード含む)の６つのクラスにtfidfを適用した結果が得られる  
* `./analysis2/all_analysis_data/all_analysis_labeled_data/ketuatu_analysis/tfidf/tfidf_data`に実行結果
* 参考サイト  
【自然言語処理】【Python】TF-IDFを使って文書の特徴をつかもう：https://zenn.dev/robes/articles/241f6c3fac1486

## 係り受け解析
* `./analysis2/all_analysis_data/all_analysis_labeled_data/ketuatu_analysis/kakariuke`にある`kakariuke.ipynb`を実行  
* 血圧が正常・高い・低い(1週間分)と血圧が正常・高い・低い(キーワード含む)の６つのクラスに係り受け解析を適用した結果が得られる  
* `./analysis2/all_analysis_data/all_analysis_labeled_data/ketuatu_analysis/kakariuke/data`に実行結果
* 参考サイト  
【実践】PythonとGiNZAで係り受け解析しようか！：https://resanaplaza.com/2022/09/11/%E3%80%90%E5%AE%9F%E8%B7%B5%E3%80%91python%E3%81%A8ginza%E3%81%A7%E4%BF%82%E3%82%8A%E5%8F%97%E3%81%91%E8%A7%A3%E6%9E%90%E3%81%97%E3%82%88%E3%81%86%E3%81%8B%EF%BC%81/

## 共起ネットワーク
ここではVisual Studio Codeではなく、Google colaboratoryで実行する  
* `./analysis2/all_analysis_data/all_analysis_labeled_data/ketuatu_analysis/kyoki_network/`に血圧が正常・高い・低い(1週間分)と血圧が正常・高い・低い(キーワード含む)の６つのクラスの各ユーザに共起ネットワークを適用した結果を可視化できるコードがある
* `./analysis2/all_analysis_data/all_analysis_labeled_data/ketuatu_analysis/kyoki_network/ketuatu_data`に各ユーザのツイートをまとめている
* 参考サイト  
【自然言語処理】【Python】共起ネットワークの作り方を理解する：https://zenn.dev/robes/articles/a3e1a6e80efd99

# 各ファイルの簡易説明
## `./analysis1/`
* `model_labeling.ipynb`  
モデルによるラベル付け
* `acc.ipynb`  
モデルの評価
* `over_cross-validation.ipynb`  
オーバーサンプリングモデル
* `user_cross-validation.ipynb`  
アンダーサンプリングモデル
* `A_topicmodel_b_undosyokuji.ipynb`  
トピックモデル結果クラスターA
* `B_topicmodel_c_syokujiseisin.ipynb`  
トピックモデル結果クラスターB
* `C_topicmodel_g_syokuji.ipynb`  
トピックモデル結果クラスターC
* `D_topicmodel_m_taicho.ipynb`  
トピックモデル結果クラスターD
* `E_topicmodel_r_seisin.ipynb`  
トピックモデル結果クラスターE

## `./analysis2/`
* `nlplot.ipynb`  
nlplotライブラリを用いた処理
* `labeling_keyword.ipynb`  
キーワードを含むツイートの特定
* `tweet_count.ipynb`  
ツイート数をカウント
* `word_count.ipynb`  
単語数をカウント
* `label_count.ipynb`  
各ラベルの出現数をカウント
* `collect_tweet_oneweek.ipynb`  
過去1週間分のツイートを収集
* `processing.ipynb`  
指標別にツイートをまとめる
* `collection_tweets.ipynb`  
3ヶ月分のツイートの収集
* `processing_csv.ipynb`  
平均ツイート数を求め、1日平均10件以上のユーザを特定
* `pre_processing.ipynb`  
数値と指標を含むツイートの特定
* `collect_indicator_tweet.ipynb`  
指標を含むツイートの特定
* `collect_profile.ipynb`  
idまたはusernameからユーザのプロフィールを取得
* `name_to_id.ipynb`  
idをusername,usernameをidに変換
* `user_iden.ipynb`  
食事・運動・体調のキーワードを含むユーザの特定
* `user_iden_profile.ipynb`  
プロフィールを参照し、ユーザを特定

# 各データの簡易説明
## `./analysis1/`
* `./analysis1/model/oversampling_model_87`  
オーバーサンプリングモデル(分類精度87%)
* `./analysis1/model/oversampling_model_77`  
アンダーサンプリングモデル(分類精度77%)
* `gakusyu_data.csv`  
モデル作成の学習データ
* `oversampling_modellabeling_87.csv`  
オーバーサンプリングモデルによる学習データへのラベル付け結果
* `undersampling_modellabeling_77.csv`  
アンダーサンプリングモデルによる学習データへのラベル付け結果
* `b_user_data.csv`  
クラスターAのツイートデータ
* `c_user_data.csv`   
クラスターBのツイートデータ
* `g_user_data.csv`  
クラスターCのツイートデータ
* `m_user_data.csv`  
クラスターDのツイートデータ
* `r_user_data.csv`  
クラスターEのツイートデータ
## `./analysis2/`  
* `pngファイル`  
各指標の高い・正常・低いツイートの上位頻出単語
* `(指標)_0_labeled.csv`  
各指標の正常値＋キーワードを含むツイート(1週間分)
* `(指標)_1_labeled.csv`  
各指標の基準値から高い＋キーワードを含むツイート(1週間分)
* `(指標)_2_labeled.csv`  
各指標の基準値から低い＋キーワードを含むツイート(1週間分)
* `(指標)_0.csv`  
各指標の正常値に関するツイート(1週間分)
* `(指標)_1.csv`  
各指標の基準値から高いに関するツイート(1週間分)
* `(指標)_2.csv`  
各指標の基準値から低いに関するツイート(1週間分)
* `stopwords.txt`  
独自のストップワード
* `analysis_username.txt`  
分析対象ユーザのusername
* `analysis.csv`  
分析対象の全ツイート(3ヶ月分)
* `index_tweet.csv`  
指標と数値を含むツイート(3ヶ月分)
* `(指標)_tweet.csv`  
各指標と数値を含むツイート(3ヶ月分)
* `all_tweets.csv`  
特定したユーザの3ヶ月分のツイート
* `final_user_id.txt`  
3ヶ月分のツイートを収集したユーザのid
* `analysis_user_id.txt`  
最終的に特定したユーザのid
* `analysis_username.txt`  
最終的に特定したユーザのusername
* `bot.txt`  
手動で発見したボットの可能性があるユーザ
* `index.csv`  
指標を含むツイート
* `final_user_info.csv`  
特定したユーザのプロフィール情報
* `final_user_twcount.csv`  
特定したユーザのツイート数
* `final_user_tweets.csv`  
特定したユーザのツイート
* `RT_parcent.csv`  
特定したユーザのRT割合
* `user_id.txt`  
キーワードを設定して収集したユーザのid
# その他参考サイト
* データの前処理：https://zenn.dev/deepblackinc/books/ad568c611643c6/viewer/c37a9f
* nlplot：https://www.takapy.work/entry/2020/05/17/192947
* scikit-learn でクラスタ分析 (K-means 法):https://pythondatascience.plavox.info/scikit-learn/%E3%82%AF%E3%83%A9%E3%82%B9%E3%82%BF%E5%88%86%E6%9E%90-k-means
* 次元削減とクラスタリング：https://qiita.com/mshinoda88/items/0e54e7e03e1aa52edb5b

# Author
* 田渕尚道
* 徳島大学大学院創成科学研究科理工学専攻
* tabuchi.n07@gmail.com
