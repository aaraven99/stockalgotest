<<<<
    with st.sidebar:
        st.markdown("### 👀 Alert Watchlist")
        if wl_tickers:
            wl_df = fetch_watchlist_status(tuple(wl_tickers))
            if not wl_df.empty:
                for _, r in wl_df.iterrows():
                    color = "#10b981" if r["Verdict"] == "STRONG BUY" else ("#ef4444" if r["Verdict"] == "STRONG SELL" else "#8b949e")
                    st.markdown(f"**{r['Ticker']}** &nbsp; <span style='color:{color};font-size:0.8rem'>{r['Verdict']}</span> &nbsp; <span style='font-size:0.85rem'>${r['Price']:.2f}</span>", unsafe_allow_html=True)
            else: st.caption("No data available for watchlist.")
        else: st.caption("Configure in Settings.")
====
    with st.sidebar:
        st.markdown("### 👀 Alert Watchlist")
        if wl_tickers:
            wl_df = fetch_watchlist_status(tuple(wl_tickers))
            if not wl_df.empty:
                for _, r in wl_df.iterrows():
                    color = "#10b981" if "BUY" in r["Verdict"] else ("#ef4444" if "SELL" in r["Verdict"] else "#8b949e")
                    st.markdown(f"**{r['Ticker']}** &nbsp; <span style='color:{color};font-size:0.8rem;font-weight:bold;'>{r['Verdict']}</span> &nbsp; <span style='font-size:0.85rem'>${r['Price']:.2f}</span>", unsafe_allow_html=True)
            else: st.caption("No data available for watchlist.")
        else: st.caption("Configure in Settings.")
>>>>
