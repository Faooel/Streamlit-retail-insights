import streamlit as st
import pandas as pd
import plotly.express as px
import os
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# --- CONFIGURATION ---
st.set_page_config(page_title="Retail Insights Dashboard", layout="wide", initial_sidebar_state="expanded")

@st.cache_data
def get_segment_stats(df):
        # Optimisation pour les graphiques page 3
        stats = df.groupby('segment')['avg_days_between_orders'].agg(['median', lambda x: x.quantile(0.8)]).reset_index()
        stats.columns = ['Segment', 'Median Days', '80% Return By']
        return stats

def load_data():
    base_path = "data/processed/"
    files = {
        "aisle": "aisle_performance.csv",
        "dept": "department_performance.csv",
        "rfm": "rfm_customer_segments.csv",
        "impulse": "impulse_products.csv",
        "enriched": "products_enriched.csv",
        "prod_segment": "products_per_segment.csv",
        "rfm_discounts": "rfm_with_discounts.csv",
        "rules_clean": "rules_clean.csv",
        "rules_dept": "rules_by_department.csv",
        "rules_seg": "rules_by_segment.csv",
        "rules_cross": "rules_cross_department.csv",
        "rules_cross_pairs": "rules_cross_department_pairs.csv",
        "rules_top": "rules_top_products.csv",
        "prod_list": "products_in_rules.csv"
    }
    
    data = {}
    for key, name in files.items():
        if os.path.exists(name):
            data[key] = pd.read_csv(name)
        elif os.path.exists(os.path.join(base_path, name)):
            data[key] = pd.read_csv(os.path.join(base_path, name))
        else:
            st.error(f"Missing file: {name}")
    return data

data = load_data()

if data and 'rfm' in data:
    df_rfm = data['rfm']
    if 'RFM_score' not in df_rfm.columns:
        df_rfm['RFM_score'] = df_rfm['R_score'] + df_rfm['F_score'] + df_rfm['M_score']
    
    # Nav
    st.sidebar.title("📊 Retail Insights")
    menu = st.sidebar.radio("Navigation", [
        "🏠 Overview", 
        "👥 Customer Segments", 
        "⏰ Purchase Timing", 
        "📦 Category Performance",
        "🍱 Smart Bundles"
    ])

# Page 1
    if menu == "🏠 Overview":
        st.title("📊 Executive Performance Overview")
        st.markdown("Global sales analysis, category performance, and revenue concentration.")

        # Segmentation filtre
        segments_list = ["All Segments"] + sorted(list(df_rfm['segment'].unique()))
        segment_selected = st.selectbox("Select a segment to filter indicators:", segments_list)

        df_filtered = df_rfm if segment_selected == "All Segments" else df_rfm[df_rfm['segment'] == segment_selected]
        
        # KPI
        total_rev = df_filtered['total_spent_eur'].sum()
        total_ord = df_filtered['num_orders'].sum()
        total_cust = len(df_filtered)
        
        aov = total_rev / total_ord if total_ord > 0 else 0
        rev_per_cust = total_rev / total_cust if total_cust > 0 else 0
        avg_days = df_filtered['avg_days_between_orders'].mean()
        avg_rfm = df_filtered['RFM_score'].mean()
        active_segments_count = len(df_filtered['segment'].unique())

        st.subheader(f"Statistics: {segment_selected}")
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Global Revenue", f"{total_rev:,.2f} €")
        m2.metric("Total Orders", f"{total_ord:,}")
        m3.metric("Unique Customers", f"{total_cust:,}")
        m4.metric("Revenue / Customer", f"{rev_per_cust:,.2f} €")

        m5, m6, m7, m8 = st.columns(4)
        m5.metric("Avg Basket (AOV)", f"{aov:.2f} €")
        m6.metric("Return Cycle", f"{avg_days:.1f} days")
        m7.metric("Avg RFM Score", f"{avg_rfm:.1f} / 15")
        m8.metric("Active Segments", active_segments_count)

        st.markdown("---")

        # Pareto graph
        if 'dept' in data:
            st.subheader("Pareto Analysis: Department Focus")
            df_pareto = data['dept'].sort_values(by='items_sold', ascending=False)
            df_pareto['cum_perc'] = 100 * (df_pareto['items_sold'].cumsum() / df_pareto['items_sold'].sum())
            
            fig_pareto = go.Figure()
            fig_pareto.add_trace(go.Bar(x=df_pareto['department'], y=df_pareto['items_sold'], name="Sales Volume", marker_color='#3b82f6'))
            fig_pareto.add_trace(go.Scatter(x=df_pareto['department'], y=df_pareto['cum_perc'], name="Cumulative %", yaxis="y2", line=dict(color="#ef4444", width=3)))
            fig_pareto.update_layout(
                yaxis2=dict(overlaying="y", side="right", range=[0, 105]), 
                height=450,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            fig_pareto.add_hline(y=80, line_dash="dash", line_color="green", annotation_text="80% Threshold", yref="y2")
            st.plotly_chart(fig_pareto, use_container_width=True)

        st.markdown("---")

        # Treemaps
        col_left, col_right = st.columns(2)
        
        with col_left:
            st.subheader("Detailed Category Performance")
            if 'dept' in data and 'aisle' in data:
                # Top departments chart
                top_15_depts = data['dept'].nlargest(15, 'items_sold')
                total_dept_items = data['dept']['items_sold'].sum()
                
                fig_dept = px.treemap(
                    top_15_depts, 
                    path=['department'], 
                    values='items_sold', 
                    color='items_sold',
                    color_continuous_scale='Blues',
                    title=f"Top 15 Departments"
                )
                fig_dept.update_layout(margin=dict(t=50, b=0, l=0, r=0), height=400)
                st.plotly_chart(fig_dept, use_container_width=True)
                
                # Aisle performance chart
                top_20_aisles = data['aisle'].nlargest(20, 'items_sold')
                total_aisle_items = data['aisle']['items_sold'].sum()
                
                fig_aisle = px.treemap(
                    top_20_aisles, 
                    path=['aisle'], 
                    values='items_sold', 
                    color='items_sold',
                    color_continuous_scale='Greys',
                    title=f"Top 20 AislesS"
                )
                fig_aisle.update_layout(margin=dict(t=50, b=0, l=0, r=0), height=500)
                st.plotly_chart(fig_aisle, use_container_width=True)

        with col_right:
            st.subheader("Revenue Concentration")
            conc = df_rfm.groupby('segment').agg(u=('user_id', 'count'), r=('total_spent_eur', 'sum')).reset_index()
            conc['% Customers'] = (conc['u'] / conc['u'].sum()) * 100
            conc['% Revenue'] = (conc['r'] / conc['r'].sum()) * 100
            
            fig_money = px.bar(
                conc, x='segment', y=['% Customers', '% Revenue'], barmode='group',
                labels={'value': 'Percentage (%)', 'variable': 'Metric'},
                color_discrete_map={'% Customers': '#94a3b8', '% Revenue': '#3b82f6'},
                title="Customer vs Revenue Concentration by Segment"
            )
            fig_money.update_layout(height=650)
            st.plotly_chart(fig_money, use_container_width=True)

# Page 2
    elif menu == "👥 Customer Segments":
        st.title("👥 Customer Segmentation & Profiling")
        st.markdown(f"**Methodology:** 8-Segment RFM Framework (Kobets & Yashyna, 2025). Total: **{len(df_rfm):,} customers**.")

        # Global segment
        col_mix1, col_mix2 = st.columns([1, 1])
        with col_mix1:
            st.subheader("Customer Base Share (%)")
            fig_pie = px.pie(df_rfm, names='segment', hole=0.4, 
                             color_discrete_sequence=px.colors.qualitative.Pastel)
            fig_pie.update_layout(margin=dict(t=30, b=0, l=0, r=0))
            st.plotly_chart(fig_pie, use_container_width=True)
            
        with col_mix2:
            st.subheader("Average RFM Profile (Scores 1-5)")
            avg_rfm = df_rfm.groupby('segment')[['R_score', 'F_score', 'M_score']].mean().reset_index()
            fig_rfm = px.bar(avg_rfm, x='segment', y=['R_score', 'F_score', 'M_score'], 
                             barmode='group', labels={'value': 'Avg Score'})
            st.plotly_chart(fig_rfm, use_container_width=True)

        st.markdown("---")

        # Srategi et top 20 produits
        st.subheader("Detailed Segment Explorer")
        selected_seg = st.selectbox("Select a segment to explore details:", sorted(df_rfm['segment'].unique()))
        
        col_info, col_prod = st.columns([1, 2])
        
        with col_info:
            st.markdown(f"#### Strategic Profile: {selected_seg}")
            df_seg = df_rfm[df_rfm['segment'] == selected_seg]
    
            st.metric("Segment Population", f"{len(df_seg):,}")
            st.metric("Avg Revenue / Customer", f"{df_seg['total_spent_eur'].mean():.2f} €")
            st.metric("Avg Orders / Customer", f"{df_seg['num_orders'].mean():.1f}")
            
            # Action Recommendation
            strategies = {
                "Premium": "VIP treatment. Focus on retention and exclusive early access.",
                "Loyal": "Reward consistency. Use cross-selling to increase basket value.",
                "Promising": "Incentivize frequency. Targeted 'next-purchase' coupons.",
                "High_Check": "Highlight high-value items and bulk-buy opportunities.",
                "New": "Onboarding journey. Introduce them to loyalty program benefits.",
                "Sleeping": "Re-activation offers with high urgency. Send 'We miss you' deals.",
                "Frugal": "Promote discounts and budget-friendly bundles.",
                "Lost": "Deep discount win-back campaigns or exit surveys."
            }
            st.warning(f"**Recommended Strategy:** \n\n {strategies.get(selected_seg, 'Standard nurturing.')}")

        with col_prod:
            st.subheader(f"Top 20 Favorite Products: {selected_seg}")
            if 'prod_segment' in data:
                prods_str = data['prod_segment'][data['prod_segment']['segment'] == selected_seg]['products'].values[0]
                prods_list = [p.strip() for p in prods_str.split(',')]

                p_col1, p_col2 = st.columns(2)
                for i, p in enumerate(prods_list[:20]):
                    if i < 10:
                        p_col1.write(f"**{i+1}.** {p}")
                    else:
                        p_col2.write(f"**{i+1}.** {p}")
            else:
                st.info("Product data for segments is not available.")

        st.markdown("---")
        st.info("**Scientific Insight:** RFM analysis transforms raw transaction data into strategic customer groups, allowing for high-precision targeting.")

# Page 3
    elif menu == "⏰ Purchase Timing":
        st.title("⏰ PURCHASE TIMING ANALYSIS")
        
        # Filtre
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            seg_filter = st.selectbox("🔍 FILTER SEGMENT:", ["All Segments"] + sorted(list(df_rfm['segment'].unique())))
        with col_f2:
            st.selectbox("🔍 FILTER CATEGORY:", ["All Categories", "Produce", "Dairy", "Beverages", "Snacks"])

        df_timing = df_rfm if seg_filter == "All Segments" else df_rfm[df_rfm['segment'] == seg_filter]

        st.markdown("---")

        st.subheader("INTER-PURCHASE TIME DISTRIBUTION")
        
        median_val = df_timing['avg_days_between_orders'].median()
        p80_val = df_timing['avg_days_between_orders'].quantile(0.8)

        plot_data = df_timing if len(df_timing) < 50000 else df_timing.sample(50000)

        fig_dist = px.histogram(
            plot_data, 
            x="avg_days_between_orders", 
            nbins=40,
            title=f"Purchase Frequency Curve - {seg_filter} (Sampled for speed)",
            labels={'avg_days_between_orders': 'Days between purchases'},
            color_discrete_sequence=['#3b82f6'],
            histnorm='probability density' 
        )
        
        fig_dist.add_vline(x=median_val, line_dash="dash", line_color="green", annotation_text=f"Med: {median_val:.1f}d")
        fig_dist.add_vline(x=p80_val, line_dash="dot", line_color="red", annotation_text=f"80%: {p80_val:.1f}d")

        fig_dist.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_dist, use_container_width=True)

        # key timing metrics
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Median Return Time", f"{median_val:.1f} days")
        with c2:
            st.metric("80% Threshold", f"{p80_val:.1f} days")
        with c3:
            long_tail = (df_timing['avg_days_between_orders'] > 90).mean() * 100
            st.metric("Inactive (90d+)", f"{long_tail:.1f}%")

        st.markdown("---")

        # segment-specific timing stats
        st.subheader("SEGMENT-SPECIFIC TIMING")
        
        df_stats = get_segment_stats(df_rfm)
        
        actions_map = {
            "Premium": "Weekly Reward", "Loyal": "Bi-weekly Promo", 
            "Promising": "Monthly Discount", "Lost": "Win-back Campaign",
            "Sleeping": "Re-activation", "New": "Onboarding", "Frugal": "Value Packs", "High_Check": "VIP Upsell"
        }
        df_stats['Action'] = df_stats['Segment'].map(actions_map).fillna("Standard")
        
        df_stats['Median Days'] = df_stats['Median Days'].map('{:,.1f}'.format)
        df_stats['80% Return By'] = df_stats['80% Return By'].map('{:,.1f} days'.format)
        
        st.table(df_stats)

        # At-risk alert
        st.subheader("AT-RISK CUSTOMER ALERTS")
        st.error(f"""
        **Strategic Advisory:** For the **{seg_filter}** group, the engagement drop-off starts after **{p80_val:.1f} days**. 
        
        **Recommendation:** To prevent churn, trigger automated triggers 2 days **before** the 80% threshold. 
        Target window: **Day {int(p80_val-2)}** (Source: Gupta & Zeithaml, 2006).
        """)

    # Page 4
    elif menu == "📦 Category Performance":
        st.title("📦 Category & Aisle Analysis")
        if 'aisle' in data:
            st.plotly_chart(px.treemap(data['aisle'].nlargest(20, 'items_sold'), path=['aisle'], values='items_sold',
                                       title="Top 20 Aisles by Volume"), use_container_width=True)

    # Page 5
    elif menu == "🍱 Smart Bundles":
        st.title("🍱 AI-Powered Smart Bundles")
        t1, t2 = st.tabs(["By Segment", "Cross-Department"])
        with t1:
            if 'rules_seg' in data:
                seg_b = st.selectbox("Target Segment:", df_rfm['segment'].unique())
                st.dataframe(data['rules_seg'][data['rules_seg']['segment'] == seg_b].nlargest(10, 'lift'), use_container_width=True)
        with t2:
            if 'rules_cross' in data:
                st.dataframe(data['rules_cross'].nlargest(10, 'lift'), use_container_width=True)

else:
    st.error("Data files not found. Please ensure all CSV files are present in the 'data/processed/' folder.")