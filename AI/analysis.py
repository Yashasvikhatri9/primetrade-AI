"""
Bitcoin Trader Performance vs Market Sentiment Analysis
Primetrade.ai — Data Science Assignment
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
import json, os

PALETTE = {
    "Extreme Fear":  "#d62728",
    "Fear":          "#ff7f0e",
    "Neutral":       "#7f7f7f",
    "Greed":         "#2ca02c",
    "Extreme Greed": "#1f77b4",
}
ORDER = ["Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"]
sns.set_theme(style="darkgrid", font_scale=1.1)
plt.rcParams.update({"figure.dpi": 140, "figure.facecolor": "#0f1117",
                     "axes.facecolor": "#1a1d27", "axes.labelcolor": "white",
                     "axes.titlecolor": "white", "xtick.color": "white",
                     "ytick.color": "white", "text.color": "white",
                     "grid.color": "#2e3250", "legend.facecolor": "#1a1d27",
                     "legend.edgecolor": "#2e3250"})

OUT = "charts"
os.makedirs(OUT, exist_ok=True)

print("Loading datasets …")
fg = pd.read_csv(r"AI\fear_greed_index - fear_greed_index.csv")
tr = pd.read_csv(r"AI\historical_data - historical_data.csv")

fg["date"] = pd.to_datetime(fg["date"])
fg = fg[["date", "value", "classification"]].rename(columns={"value": "fg_value"})

tr["date"] = pd.to_datetime(tr["Timestamp IST"], dayfirst=True, errors="coerce").dt.normalize()
tr = tr.dropna(subset=["date"])
tr["Closed PnL"]  = pd.to_numeric(tr["Closed PnL"],  errors="coerce").fillna(0)
tr["Size USD"]    = pd.to_numeric(tr["Size USD"],     errors="coerce").fillna(0)
tr["Fee"]         = pd.to_numeric(tr["Fee"],          errors="coerce").fillna(0)
tr["is_close"]    = tr["Closed PnL"] != 0           
tr["is_win"]      = tr["Closed PnL"] > 0

df = tr.merge(fg, on="date", how="inner")
print(f"  Trades after merge: {len(df):,}  |  date range: {df['date'].min().date()} → {df['date'].max().date()}")

closes = df[df["is_close"]].copy()
print(f"  Closing trades:     {len(closes):,}")


def sentiment_stats(data):
    g = data.groupby("classification")
    stats_df = pd.DataFrame({
        "total_trades":   g["Closed PnL"].count(),
        "total_pnl":      g["Closed PnL"].sum(),
        "avg_pnl":        g["Closed PnL"].mean(),
        "median_pnl":     g["Closed PnL"].median(),
        "win_rate":       g["is_win"].mean() * 100,
        "total_volume":   g["Size USD"].sum(),
        "avg_volume":     g["Size USD"].mean(),
        "total_fee":      g["Fee"].sum(),
    }).round(4)
    stats_df = stats_df.reindex([s for s in ORDER if s in stats_df.index])
    return stats_df

sent_stats = sentiment_stats(closes)
print("\nSentiment Stats:\n", sent_stats.to_string())

trader_sent = (
    closes.groupby(["Account", "classification"])
    .agg(trades=("Closed PnL", "count"),
         total_pnl=("Closed PnL", "sum"),
         win_rate=("is_win", "mean"),
         avg_pnl=("Closed PnL", "mean"))
    .reset_index()
)

daily = (
    closes.groupby(["date", "classification", "fg_value"])
    .agg(daily_pnl=("Closed PnL", "sum"),
         trades=("Closed PnL", "count"),
         win_rate=("is_win", "mean"))
    .reset_index()
)

coin_sent = (
    closes.groupby(["Coin", "classification"])
    .agg(total_pnl=("Closed PnL", "sum"),
         trades=("Closed PnL", "count"),
         win_rate=("is_win", "mean"))
    .reset_index()
)
side_sent = (
    df.groupby(["classification", "Side"])
    .size().reset_index(name="count")
)


colors_ord = [PALETTE[s] for s in ORDER if s in sent_stats.index]
cats       = [s for s in ORDER if s in sent_stats.index]

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(cats, sent_stats.loc[cats, "win_rate"], color=colors_ord,
              width=0.55, edgecolor="#ffffff22", linewidth=0.8)
for bar, val in zip(bars, sent_stats.loc[cats, "win_rate"]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.4,
            f"{val:.1f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")
ax.set_title("Win Rate by Market Sentiment", fontsize=14, fontweight="bold", pad=14)
ax.set_ylabel("Win Rate (%)")
ax.set_ylim(0, 75)
ax.axhline(50, color="white", linewidth=0.8, linestyle="--", alpha=0.5, label="50% baseline")
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUT}/1_win_rate_by_sentiment.png", bbox_inches="tight")
plt.close()
print("Saved chart 1")


fig, ax = plt.subplots(figsize=(10, 5))
vals = sent_stats.loc[cats, "avg_pnl"]
bar_colors = ["#d62728" if v < 0 else c for v, c in zip(vals, colors_ord)]
bars = ax.bar(cats, vals, color=bar_colors, width=0.55, edgecolor="#ffffff22")
for bar, val in zip(bars, vals):
    ypos = bar.get_height() + (1 if val >= 0 else -3)
    ax.text(bar.get_x() + bar.get_width()/2, ypos,
            f"${val:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
ax.axhline(0, color="white", linewidth=0.8, linestyle="--", alpha=0.5)
ax.set_title("Average Closed PnL per Trade by Sentiment", fontsize=14, fontweight="bold", pad=14)
ax.set_ylabel("Avg PnL (USD)")
plt.tight_layout()
plt.savefig(f"{OUT}/2_avg_pnl_by_sentiment.png", bbox_inches="tight")
plt.close()
print("Saved chart 2")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax1 = axes[0]
vals = sent_stats.loc[cats, "total_pnl"] / 1e6
bar_colors = ["#d62728" if v < 0 else c for v, c in zip(vals, colors_ord)]
bars = ax1.bar(cats, vals, color=bar_colors, width=0.55, edgecolor="#ffffff22")
for bar, val in zip(bars, vals):
    ax1.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + (0.005 if val >= 0 else -0.03),
             f"${val:.2f}M", ha="center", va="bottom", fontsize=9, fontweight="bold")
ax1.axhline(0, color="white", linewidth=0.8, linestyle="--", alpha=0.5)
ax1.set_title("Total Closed PnL by Sentiment", fontsize=13, fontweight="bold")
ax1.set_ylabel("Total PnL (USD Millions)")

ax2 = axes[1]
vols = sent_stats.loc[cats, "total_volume"] / 1e9
bars2 = ax2.bar(cats, vols, color=colors_ord, width=0.55, edgecolor="#ffffff22")
for bar, val in zip(bars2, vols):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
             f"${val:.2f}B", ha="center", va="bottom", fontsize=9, fontweight="bold")
ax2.set_title("Total Trading Volume by Sentiment", fontsize=13, fontweight="bold")
ax2.set_ylabel("Volume (USD Billions)")

plt.suptitle("PnL & Volume Distribution Across Sentiment Regimes", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(f"{OUT}/3_pnl_and_volume.png", bbox_inches="tight")
plt.close()
print("Saved chart 3")

fig, ax = plt.subplots(figsize=(12, 6))
valid = closes[closes["classification"].isin(ORDER) & (closes["Closed PnL"].between(-2000, 2000))]
parts = ax.violinplot(
    [valid[valid["classification"] == s]["Closed PnL"].values for s in ORDER],
    positions=range(len(ORDER)), widths=0.6, showmedians=True, showextrema=True
)
for i, (pc, s) in enumerate(zip(parts["bodies"], ORDER)):
    pc.set_facecolor(PALETTE[s])
    pc.set_alpha(0.7)
parts["cmedians"].set_color("white")
parts["cmedians"].set_linewidth(2)
parts["cbars"].set_color("white")
parts["cmins"].set_color("white")
parts["cmaxes"].set_color("white")
ax.set_xticks(range(len(ORDER)))
ax.set_xticklabels(ORDER)
ax.axhline(0, color="white", linewidth=0.8, linestyle="--", alpha=0.5)
ax.set_title("PnL Distribution by Market Sentiment (clipped ±$2,000)", fontsize=14, fontweight="bold", pad=14)
ax.set_ylabel("Closed PnL (USD)")
plt.tight_layout()
plt.savefig(f"{OUT}/4_pnl_violin.png", bbox_inches="tight")
plt.close()
print("Saved chart 4")

fig, ax = plt.subplots(figsize=(11, 5))
pivot_side = side_sent.pivot(index="classification", columns="Side", values="count").fillna(0)
pivot_side = pivot_side.reindex([s for s in ORDER if s in pivot_side.index])
total = pivot_side.sum(axis=1)
buy_pct  = pivot_side.get("BUY",  0) / total * 100
sell_pct = pivot_side.get("SELL", 0) / total * 100
x = np.arange(len(pivot_side))
w = 0.4
ax.bar(x - w/2, buy_pct,  w, label="BUY",  color="#2ca02c", edgecolor="#ffffff22")
ax.bar(x + w/2, sell_pct, w, label="SELL", color="#d62728", edgecolor="#ffffff22")
ax.axhline(50, color="white", linewidth=0.8, linestyle="--", alpha=0.5)
ax.set_xticks(x)
ax.set_xticklabels(pivot_side.index)
ax.set_ylabel("% of Trades")
ax.set_title("Buy vs Sell Distribution by Market Sentiment", fontsize=14, fontweight="bold", pad=14)
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUT}/5_buy_sell_ratio.png", bbox_inches="tight")
plt.close()
print("Saved chart 5")

top_coins = closes.groupby("Coin")["Closed PnL"].sum().nlargest(10).index.tolist()
coin_pivot = (
    coin_sent[coin_sent["Coin"].isin(top_coins)]
    .pivot(index="Coin", columns="classification", values="total_pnl")
    .fillna(0)
)
coin_pivot = coin_pivot.reindex(columns=[s for s in ORDER if s in coin_pivot.columns])

fig, ax = plt.subplots(figsize=(12, 7))
sns.heatmap(coin_pivot / 1000, annot=True, fmt=".1f", cmap="RdYlGn",
            linewidths=0.5, linecolor="#0f1117", ax=ax,
            cbar_kws={"label": "PnL (USD Thousands)"})
ax.set_title("Top 10 Coins — Total PnL by Sentiment (USD K)", fontsize=14, fontweight="bold", pad=14)
ax.set_xlabel("Sentiment")
ax.set_ylabel("Coin")
plt.tight_layout()
plt.savefig(f"{OUT}/6_coin_sentiment_heatmap.png", bbox_inches="tight")
plt.close()
print("Saved chart 6")


fig, ax = plt.subplots(figsize=(11, 6))
for sent in ORDER:
    sub = daily[daily["classification"] == sent]
    ax.scatter(sub["fg_value"], sub["daily_pnl"] / 1000,
               color=PALETTE[sent], alpha=0.6, s=25, label=sent)
m, b, r, p, _ = stats.linregress(daily["fg_value"], daily["daily_pnl"])
x_line = np.linspace(daily["fg_value"].min(), daily["fg_value"].max(), 100)
ax.plot(x_line, (m * x_line + b) / 1000, color="white", linewidth=1.5,
        linestyle="--", label=f"Trend (r={r:.2f}, p={p:.3f})")
ax.axhline(0, color="white", linewidth=0.5, alpha=0.4)
ax.set_xlabel("Fear & Greed Index Value (0=Extreme Fear, 100=Extreme Greed)")
ax.set_ylabel("Daily Aggregate PnL (USD Thousands)")
ax.set_title("Fear-Greed Score vs Daily Trader PnL", fontsize=14, fontweight="bold", pad=14)
ax.legend(markerscale=1.2)
plt.tight_layout()
plt.savefig(f"{OUT}/7_fg_vs_daily_pnl.png", bbox_inches="tight")
plt.close()
print("Saved chart 7")

daily_count = df.groupby(["date", "classification"]).size().reset_index(name="count")
fig, ax = plt.subplots(figsize=(14, 5))
for sent in ORDER:
    sub = daily_count[daily_count["classification"] == sent]
    ax.scatter(sub["date"], sub["count"], color=PALETTE[sent],
               alpha=0.7, s=15, label=sent)
ax.set_title("Daily Trade Activity Coloured by Sentiment", fontsize=14, fontweight="bold", pad=14)
ax.set_ylabel("Trades per Day")
ax.set_xlabel("Date")
ax.legend(markerscale=1.5, ncol=5)
plt.tight_layout()
plt.savefig(f"{OUT}/8_daily_trade_activity.png", bbox_inches="tight")
plt.close()
print("Saved chart 8")

top10 = (closes.groupby("Account")["Closed PnL"].sum()
         .nlargest(10).reset_index())
top10["label"] = top10["Account"].str[:8] + "…"
fig, ax = plt.subplots(figsize=(12, 6))
bars = ax.barh(top10["label"][::-1], top10["Closed PnL"][::-1] / 1000,
               color="#1f77b4", edgecolor="#ffffff22")
for bar, val in zip(bars, top10["Closed PnL"][::-1] / 1000):
    ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
            f"${val:.1f}K", va="center", fontsize=9)
ax.set_xlabel("Total Closed PnL (USD Thousands)")
ax.set_title("Top 10 Traders by Total Closed PnL", fontsize=14, fontweight="bold", pad=14)
plt.tight_layout()
plt.savefig(f"{OUT}/9_top10_traders.png", bbox_inches="tight")
plt.close()
print("Saved chart 9")

fig = plt.figure(figsize=(16, 9))
fig.patch.set_facecolor("#0f1117")
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

axA = fig.add_subplot(gs[0, 0])
axA.bar(cats, sent_stats.loc[cats, "win_rate"], color=colors_ord, width=0.6, edgecolor="#ffffff22")
axA.axhline(50, color="white", lw=0.8, ls="--", alpha=0.5)
axA.set_title("Win Rate (%)", fontsize=11, fontweight="bold")
axA.set_facecolor("#1a1d27")

axB = fig.add_subplot(gs[0, 1])
vals_b = sent_stats.loc[cats, "avg_pnl"]
axB.bar(cats, vals_b, color=["#d62728" if v < 0 else c for v, c in zip(vals_b, colors_ord)],
        width=0.6, edgecolor="#ffffff22")
axB.axhline(0, color="white", lw=0.8, ls="--", alpha=0.5)
axB.set_title("Avg PnL / Trade ($)", fontsize=11, fontweight="bold")
axB.set_facecolor("#1a1d27")

axC = fig.add_subplot(gs[0, 2])
axC.bar(cats, sent_stats.loc[cats, "total_trades"], color=colors_ord, width=0.6, edgecolor="#ffffff22")
axC.set_title("Total Closing Trades", fontsize=11, fontweight="bold")
axC.set_facecolor("#1a1d27")

axD = fig.add_subplot(gs[1, 0])
tp = sent_stats.loc[cats, "total_pnl"] / 1e6
axD.bar(cats, tp, color=["#d62728" if v < 0 else c for v, c in zip(tp, colors_ord)],
        width=0.6, edgecolor="#ffffff22")
axD.axhline(0, color="white", lw=0.8, ls="--", alpha=0.5)
axD.set_title("Total PnL ($ Millions)", fontsize=11, fontweight="bold")
axD.set_facecolor("#1a1d27")

axE = fig.add_subplot(gs[1, 1])
axE.bar(cats, sent_stats.loc[cats, "total_volume"] / 1e9, color=colors_ord, width=0.6, edgecolor="#ffffff22")
axE.set_title("Volume ($ Billions)", fontsize=11, fontweight="bold")
axE.set_facecolor("#1a1d27")

axF = fig.add_subplot(gs[1, 2])
axF.bar(cats, sent_stats.loc[cats, "total_fee"] / 1e3, color=colors_ord, width=0.6, edgecolor="#ffffff22")
axF.set_title("Total Fees ($ Thousands)", fontsize=11, fontweight="bold")
axF.set_facecolor("#1a1d27")

for ax in [axA, axB, axC, axD, axE, axF]:
    ax.tick_params(axis="x", rotation=22, labelsize=8)
    ax.tick_params(axis="y", labelsize=8)

fig.suptitle("Trader Performance Dashboard — by Market Sentiment",
             fontsize=16, fontweight="bold", color="white", y=1.01)
plt.savefig(f"{OUT}/10_summary_dashboard.png", bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print("Saved chart 10")

sent_stats.to_csv("sentiment_stats.csv")
trader_sent.to_csv("trader_sentiment_breakdown.csv", index=False)

best_sent     = sent_stats["win_rate"].idxmax()
worst_sent    = sent_stats["win_rate"].idxmin()
highest_pnl   = sent_stats["total_pnl"].idxmax()
lowest_pnl    = sent_stats["total_pnl"].idxmin()
highest_vol   = sent_stats["total_volume"].idxmax()

insights = {
    "dataset_summary": {
        "total_trades": int(len(df)),
        "closing_trades": int(len(closes)),
        "unique_traders": int(df["Account"].nunique()),
        "unique_coins": int(df["Coin"].nunique()),
        "date_range": f"{df['date'].min().date()} to {df['date'].max().date()}"
    },
    "key_findings": {
        "best_win_rate_sentiment": best_sent,
        "best_win_rate_value": round(float(sent_stats.loc[best_sent, "win_rate"]), 2),
        "worst_win_rate_sentiment": worst_sent,
        "worst_win_rate_value": round(float(sent_stats.loc[worst_sent, "win_rate"]), 2),
        "highest_total_pnl_sentiment": highest_pnl,
        "lowest_total_pnl_sentiment": lowest_pnl,
        "highest_volume_sentiment": highest_vol,
        "fg_pnl_correlation_r": round(float(stats.pearsonr(daily["fg_value"], daily["daily_pnl"])[0]), 4),
        "fg_pnl_correlation_p": round(float(stats.pearsonr(daily["fg_value"], daily["daily_pnl"])[1]), 4),
    },
    "sentiment_stats": sent_stats.reset_index().to_dict(orient="records")
}

with open("insights.json", "w") as f:
    json.dump(insights, f, indent=2)

print("\n All charts and stats saved.")
print(json.dumps(insights["key_findings"], indent=2))
