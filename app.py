import streamlit as st
import pandas as pd
import plotly.express as px
import os
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import json

# Config
st.set_page_config(page_title="Retail insights dashboard", layout="wide", initial_sidebar_state="expanded")

# Cached data loading
@st.cache_data
def load_data():
    base_path = "data/processed/"
    files = {
        "aisle": "aisle_performance.csv",
        "dept": "department_performance.csv",
        "rfm": "rfm_customer_segments.csv",
        "enriched": "products_enriched.csv",
        "prod_segment": "products_per_segment.csv",
        "rfm_discounts": "rfm_with_discounts.csv",
        "rules_clean": "rules_clean.csv",
        "rules_dept": "rules_by_department.csv",
        "rules_seg": "rules_by_segment.csv",
        "rules_cross": "rules_cross_department.csv",
        "rules_cross_pairs": "rules_cross_department_pairs.csv",
        "rules_top": "rules_top_products.csv",
        "financial": "financial_impact_summary.csv",
        "segment_metrics": "segment_metrics.csv",
        "bundle_rec": "bundle_recommendations.csv",
        "promo_efficiency": "promotion_efficiency.csv",
        "sensitivity": "sensitivity_analysis.csv",
        "metadata": "metadata_06.json",
        "departments": "departments_optimized.csv",
        "aisles": "aisles_optimized.csv"
    }
    
    data = {}
    for key, name in files.items():
        path = os.path.join(base_path, name)
        if os.path.exists(path):
            if key == "metadata":
                with open(path, 'r') as f:
                    data[key] = json.load(f)
            else:
                data[key] = pd.read_csv(path)
        else:
            st.warning(f"File missing: {name}")
    
    return data

# Helper functions
@st.cache_data
def get_segment_stats(df):
    stats = df.groupby('segment')['avg_days_between_orders'].agg(['median', lambda x: x.quantile(0.8)]).reset_index()
    stats.columns = ['Segment', 'Median days', '80% return by']
    return stats

# Main app
def main():
    data = load_data()
    
    st.sidebar.title("🛒 Retail insights")
    menu = st.sidebar.radio(
        "Navigation",
        [
            "🏠 Home",
            "💰 Financial impact",
            "👥 Customer segments",
            "⏰ Purchase timing",
            "📦 Category performance",
            "🍱 Smart bundles",
            "📈 Scenario planner",
            "ℹ️ Sources"
        ]
    )
    
   
    # Homepage
   
    if menu == "🏠 Home":
        st.title("🏠 Retail insights dashboard")
        st.markdown("**Data-driven retail insights for cost savings & revenue growth**")
        st.markdown("*DSTI Project 2 - Applied MSc in Data Analytics/Science/Engineering*")
        
        if "metadata" in data:
            meta = data["metadata"]
            stats = meta.get("summary_statistics", {})
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total customers", f"{stats.get('total_customers', 0):,}")
            with col2:
                st.metric("Annual net impact", f"EUR {stats.get('total_net_impact_eur', 0):,.0f}", 
                         delta=f"+{stats.get('average_roi', 0)}x ROI")
            with col3:
                st.metric("Targeting efficiency", f"{stats.get('roi_improvement', 0)}x", 
                         delta="vs. untargeted")
            with col4:
                st.metric("Customer coverage", f"{stats.get('customer_coverage_pct', 0):.1f}%", 
                         delta="30-60% target")
            
            st.info("""
            💡 **What is net impact?**
            Estimated additional annual profit from targeted promotions:
            - Revenue lift from increased purchases
            - Plus bundle upsell revenue
            - Minus discount costs
            
            *Based on research-backed lift rates (Wamsler et al., 2024)*
            """)
        
        if "segment_metrics" in data:
            st.subheader("Annual net impact by customer segment")
            df_seg = data["segment_metrics"].sort_values("net_impact_eur", ascending=False)
            
            fig = px.bar(
                df_seg,
                x="net_impact_eur",
                y="segment",
                orientation="h",
                color="net_impact_eur",
                color_continuous_scale="Blues",
                labels={"net_impact_eur": "Net impact (EUR)", "segment": "Customer segment"}
            )
            fig.update_layout(xaxis_title="Net impact (EUR)", yaxis_title="", showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            
            st.caption("""
            🔍 **Key insight**: Premium and loyal customers drive the highest net impact. 
            Focus promotions on these segments first for maximum return.
            """)
        
        if "rfm" in data:
            df_rfm = data["rfm"]
            st.subheader("Quick stats")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Avg revenue / customer", f"{df_rfm['total_spent_eur'].mean():.2f} €")
            with c2:
                st.metric("Avg orders / customer", f"{df_rfm['num_orders'].mean():.1f}")
            with c3:
                st.metric("Avg days between orders", f"{df_rfm['avg_days_between_orders'].median():.1f} days")
    
   
    # Financial impact
   
    elif menu == "💰 Financial impact":
        st.title("💰 Financial impact analysis")
        
        if "financial" in data:
            df_fin = data["financial"]
            
            selected_seg = st.selectbox("Select segment", sorted(df_fin["segment"].unique()))
            seg_fin = df_fin[df_fin["segment"] == selected_seg].iloc[0] if len(df_fin[df_fin["segment"] == selected_seg]) > 0 else None
            
            if seg_fin is not None:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Net impact", f"€{seg_fin.get('net_impact_eur', 0):,.0f}")
                with col2:
                    roi_val = seg_fin.get("roi", "N/A")
                    st.metric("ROI", f"{roi_val}x" if roi_val != "N/A" else "N/A")
                with col3:
                    roi_imp = seg_fin.get("roi_improvement_factor", "N/A")
                    st.metric("ROI improvement", f"{roi_imp}x" if roi_imp != "N/A" else "N/A")
                
                st.subheader("Financial impact breakdown")
                waterfall_data = pd.DataFrame({
                    "Component": ["Baseline revenue", "Revenue lift (+42.3%)", "Bundle upsell", "Discount cost", "Net impact"],
                    "Value": [
                        seg_fin.get("baseline_revenue_eur", 0),
                        seg_fin.get("revenue_lift_eur", 0),
                        seg_fin.get("bundle_uplift_eur", 0),
                        -seg_fin.get("discount_cost_eur", 0),
                        seg_fin.get("net_impact_eur", 0)
                    ]
                })
                
                fig = go.Figure(go.Waterfall(   
                x=waterfall_data["Component"],
                y=waterfall_data["Value"],
                connector_mode="spanning"  # <--- Correction effectuée
                ))
                fig.update_layout(title=f"Financial impact breakdown: {selected_seg}", yaxis_title="EUR")
                st.plotly_chart(fig, use_container_width=True)
                
                st.info("""
                📊 **How to read this chart**:
                1. **Baseline revenue**: What this segment spends today
                2. **Revenue lift (+42.3%)**: Extra revenue from increased purchase frequency
                3. **Bundle upsell**: Extra revenue from cross-sell recommendations
                4. **Discount cost**: Money spent on discounts (only paid when redeemed)
                5. **Net impact**: Final profit impact (Lift + Upsell − Cost)
                """)
            
            if "promo_efficiency" in data:
                st.subheader("Targeted vs. untargeted promotion ROI")
                df_eff = data["promo_efficiency"]
                
                df_melt = df_eff.melt(
                    id_vars="segment",
                    value_vars=["roi", "untargeted_roi"],
                    var_name="Approach",
                    value_name="ROI"
                )
                df_melt = df_melt[df_melt["ROI"] != "N/A"]
                df_melt["ROI"] = pd.to_numeric(df_melt["ROI"], errors="coerce")
                df_melt["Approach"] = df_melt["Approach"].map({"roi": "Targeted", "untargeted_roi": "Untargeted"})
                
                fig = px.bar(
                    df_melt,
                    x="segment",
                    y="ROI",
                    color="Approach",
                    barmode="group",
                    title="Targeted vs. untargeted promotion ROI by segment",
                    labels={"ROI": "ROI (EUR returned per EUR spent)", "segment": "Segment"}
                )
                st.plotly_chart(fig, use_container_width=True)
                
                st.success("""
                ✅ **Why targeting works better**:
                - **Higher redemption**: 42.4% vs. 27.7% (customers redeem when timing matches their needs)
                - **Higher revenue lift**: Targeted promotions trigger +42.3% purchase frequency
                - **Better bundle adoption**: Relevant recommendations convert better
                
                *Result: Every €1 spent on targeted discounts generates more net impact than untargeted promotions.*
                """)
    
   
    # Customer segments
   
    elif menu == "👥 Customer segments":
        st.title("👥 Customer segmentation")
        st.write(f"Based on RFM analysis. Total customers: **{len(data.get('rfm', [])):,}**")
        
        if "rfm" in data:
            df_rfm = data["rfm"]
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Customer share (%)")
                fig_pie = px.pie(df_rfm, names='segment', hole=0.4, color_discrete_sequence=px.colors.qualitative.Pastel)
                st.plotly_chart(fig_pie, use_container_width=True)
            with col2:
                st.subheader("Average RFM scores")
                avg_rfm = df_rfm.groupby('segment')[['R_score', 'F_score', 'M_score']].mean().reset_index()
                fig_rfm = px.bar(avg_rfm, x='segment', y=['R_score', 'F_score', 'M_score'], barmode='group', labels={'value': 'Score (1-5)'})
                st.plotly_chart(fig_rfm, use_container_width=True)
            
            st.markdown("-")
            st.subheader("Segment explorer")
            selected_seg = st.selectbox("Select a group to analyze:", sorted(df_rfm['segment'].unique()))
            
            col_left, col_right = st.columns([1, 2])
            with col_left:
                st.markdown(f"### Profile: {selected_seg}")
                df_seg = df_rfm[df_rfm['segment'] == selected_seg]
                st.metric("Total customers", f"{len(df_seg):,}")
                st.metric("Avg revenue", f"{df_seg['total_spent_eur'].mean():.2f} €")
                st.metric("Avg orders", f"{df_seg['num_orders'].mean():.1f}")
                
                actions = {
                    "Premium": "VIP loyalty rewards and exclusive early access.",
                    "Loyal": "Retention focus. Use cross-selling to increase basket size.",
                    "Promising": "Incentivize frequency with 'next-purchase' coupons.",
                    "High_Check": "Highlight premium items and bulk deals.",
                    "New": "Onboarding journey. Introduce them to loyalty benefits.",
                    "Sleeping": "Re-activation offers. Send urgent 'We miss you' deals.",
                    "Frugal": "Target with discounts and budget bundles.",
                    "Lost": "Aggressive win-back campaigns or exit surveys."
                }
                st.info(f"**Action:** {actions.get(selected_seg, 'Standard marketing.')}")
            
            with col_right:
                st.subheader(f"Top favorite products")
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
                    st.info("Product list not available.")
            
            st.markdown("-")
            st.write("**Insight:** RFM segments (Kobets & Yashyna 2025) help target customers with the right message at the right time.")
    
   
    # Purchase timing
   
    elif menu == "⏰ Purchase timing":
        st.title("⏰ Purchase frequency analysis")
        st.write("Analysis of customer return cycles and churn risk thresholds.")
        
        if "rfm" in data:
            df_rfm = data["rfm"]
            
            col_f1, col_f2 = st.columns(2)
            with col_f1:
                segments = ["All segments"] + sorted(list(df_rfm['segment'].unique()))
                selected_seg = st.selectbox("Filter by segment:", segments)
            with col_f2:
                categories = ["All categories", "Produce", "Dairy", "Beverages"]
                selected_cat = st.selectbox("Filter by category:", categories)
            
            if selected_seg == "All segments":
                df_plot = df_rfm
            else:
                df_plot = df_rfm[df_rfm['segment'] == selected_seg]
            
            median_days = df_plot['avg_days_between_orders'].median()
            p80_days = df_plot['avg_days_between_orders'].quantile(0.8)
            k1, k2 = st.columns(2)
            k1.metric("Median return time", f"{median_days:.1f} days")
            k2.metric("80% return threshold", f"{p80_days:.1f} days")
            st.markdown("-")
            
            df_sample = df_plot.sample(min(len(df_plot), 25000))
            fig = px.histogram(
                df_sample,
                x="avg_days_between_orders",
                nbins=50,
                title=f"Return cycle: {selected_seg} | {selected_cat}",
                color_discrete_sequence=['#3b82f6'],
                labels={'avg_days_between_orders': 'Days'}
            )
            fig.add_vline(x=p80_days, line_dash="dot", line_color="red", annotation_text="80% limit")
            st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("Segment benchmarks & recommended actions")
            timing_table = get_segment_stats(df_rfm)
            
            actions = {
                "Premium": "VIP reward program",
                "Loyal": "Cross-sell promotion",
                "Promising": "Next-purchase discount",
                "High_Check": "Bulk buy incentives",
                "New": "Welcome coupon",
                "Sleeping": "Re-activation email",
                "Frugal": "Value pack deals",
                "Lost": "Win-back campaign"
            }
            timing_table['Recommended action'] = timing_table['Segment'].map(actions)
            timing_table['Median days'] = timing_table['Median days'].map('{:.1f}'.format)
            timing_table['80% return by'] = timing_table['80% return by'].map('{:.1f} days'.format)
            st.table(timing_table)
            
            st.warning(f"**Retention strategy:** For {selected_seg}, customers are likely to churn after **{int(p80_days)} days**. Target them at day **{int(p80_days - 2)}**.")
    
   
    # Category performance
   
    elif menu == "📦 Category performance":
        st.title("📦 Category & shopping journey analysis")
        st.write("Analysis of product roles: Anchors (triggers) vs last-position (complements).")
        
        if 'aisle' in data:
            st.subheader("Top 20 aisles by volume")
            top_aisles = data['aisle'].nlargest(20, 'items_sold')
            fig_aisle = px.treemap(
                top_aisles,
                path=['aisle'],
                values='items_sold',
                color='items_sold',
                color_continuous_scale='Blues'
            )
            fig_aisle.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=450)
            st.plotly_chart(fig_aisle, use_container_width=True)
            
            st.markdown("-")
            
            st.subheader("Shopping journey analysis")
            st.caption("Note: Ratio = sur-representation ratio in a specific position compared to the global average.")
            
            col_left, col_right = st.columns(2)
            with col_left:
                st.info("### ⚓ Anchor products (first in cart)")
                if 'first_pos' in data:
                    top_anchors = data['first_pos'].nlargest(10, 'first3_ratio')
                    fig_anchors = px.bar(
                        top_anchors,
                        x='first3_ratio',
                        y='product_name',
                        orientation='h',
                        color='first3_ratio',
                        color_continuous_scale='Blues',
                        title="Top 10: first_position_ratio",
                        labels={'first3_ratio': 'first_position_ratio', 'product_name': ''}
                    )
                    fig_anchors.update_layout(yaxis={'categoryorder':'total ascending'})
                    st.plotly_chart(fig_anchors, use_container_width=True)
            
            with col_right:
                st.info("### 🛒 Complement products (last in cart)")
                if 'last_pos' in data:
                    top_lasts = data['last_pos'].nlargest(10, 'last_position_ratio')
                    fig_lasts = px.bar(
                        top_lasts,
                        x='last_position_ratio',
                        y='product_name',
                        orientation='h',
                        color='last_position_ratio',
                        color_continuous_scale='Oranges',
                        title="Top 10: last_position_ratio",
                        labels={'last_position_ratio': 'last_position_ratio', 'product_name': ''}
                    )
                    fig_lasts.update_layout(yaxis={'categoryorder':'total ascending'})
                    st.plotly_chart(fig_lasts, use_container_width=True)
            
            st.markdown("-")
            st.subheader("Retail strategy suggestions")
            rec_col1, rec_col2 = st.columns(2)
            with rec_col1:
                st.success("**Navigation:** Place high *first_position_ratio* items (anchors) at the back of the store to increase travel distance.")
            with rec_col2:
                st.error("**Checkout:** Use high *last_position_ratio* items for checkout displays or final app notifications.")
    
   
    # Smart bundles
   
    elif menu == "🍱 Smart bundles":
        st.title("🍱 Product bundles & association rules")
        st.write("Identify product pairings to increase average order value (AOV) based on transaction history.")
        
        tab_seg, tab_dept, tab_cross = st.tabs(["By segment", "By department", "Cross-department"])
        
        with tab_seg:
            st.subheader("Segment-specific rules")
            if 'bundle_rec' in data:
                target_seg = st.selectbox("Select target segment", sorted(data['bundle_rec']['segment'].unique()))
                rules_s = data['bundle_rec'][data['bundle_rec']['segment'] == target_seg].sort_values("estimated_revenue_per_recommendation", ascending=False).head(10)
                
                if not rules_s.empty:
                    for idx, row in rules_s.iterrows():
                        with st.container(border=True):
                            col1, col2, col3 = st.columns([2, 1, 1])
                            with col1:
                                st.subheader(f"{str(row['antecedent'])[:50]} → {str(row['consequent'])[:50]}")
                                st.caption(f"{row['antecedent_dept']} → {row['consequent_dept']}")
                            with col2:
                                st.metric("Est. revenue", f"€{row['estimated_revenue_per_recommendation']:.2f}")
                                st.caption(f"Lift: {row['lift']:.2f}x")
                            with col3:
                                st.metric("Margin", f"{row['gross_margin_used']*100:.0f}%")
                                st.caption(f"Discount: {row['recommended_discount_percent']:.1f}%")
                else:
                    st.info("No strong associations found for this segment.")
        
        with tab_dept:
            st.subheader("Intra-department pairings")
            if 'rules_dept' in data:
                depts = sorted(data['rules_dept']['department'].unique())
                target_dept = st.selectbox("Select department:", depts)
                dept_rules = data['rules_dept'][data['rules_dept']['department'] == target_dept].sort_values('lift', ascending=False).head(10)
                st.dataframe(dept_rules[['antecedent', 'consequent', 'confidence', 'lift']], use_container_width=True)
        
        with tab_cross:
            st.subheader("Cross-department sales")
            st.write("Pairings between different store areas.")
            if 'rules_cross_pairs' in data:
                cross_rules = data['rules_cross_pairs'].sort_values('lift', ascending=False).head(15)
                st.dataframe(cross_rules[['antecedent', 'consequent', 'antecedent_dept', 'consequent_dept', 'lift']], use_container_width=True)
                st.write("**Note:** These pairs are ideal for cross-aisle promotions or 'bundle' kits.")
        
        with st.expander("📚 Understanding metrics"):
            st.markdown("""
            - **Confidence:** Probability that the *consequent* is bought when the *antecedent* is in the cart.
            - **Lift:** The strength of the association. A lift > 1 means the items are bought together much more often than by random chance.
            - **Gross margin:** Department-specific profit margin before operating expenses (sourced from industry reports).
            - **Est. revenue:** Expected extra revenue per bundle (basket size × lift × margin).
            """)
    
   
    # Scenario planner
   
    elif menu == "📈 Scenario planner":
        st.title("📈 Scenario planner")
        st.markdown("**What-if analysis:** Explore different scenarios for promotion performance")
        
        if "sensitivity" in data:
            df_sens = data["sensitivity"]
            
            scenario = st.radio("Select scenario", ["conservative", "baseline", "optimistic"], horizontal=True)
            df_scenario = df_sens[df_sens["scenario"] == scenario]
            
            col1, col2, col3 = st.columns(3)
            scenarios = ["conservative", "baseline", "optimistic"]
            for scen, col in zip(scenarios, [col1, col2, col3]):
                scen_data = df_sens[df_sens["scenario"] == scen]
                total_impact = scen_data["net_impact_eur"].sum()
                with col:
                    st.metric(scen.capitalize(), f"€{total_impact:,.0f}")
            
            fig = px.bar(
                df_sens,
                x="scenario",
                y="net_impact_eur",
                color="scenario",
                labels={"net_impact_eur": "Net impact (EUR)", "scenario": "Scenario"},
                title="Net impact range by scenario"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.info("""
            📊 **What do these scenarios mean?**
            - **Conservative:** Lower redemption (30%), lower margins, lower adoption
            - **Baseline:** Research-backed assumptions (42.4% redemption, 30% margin, 12% adoption)
            - **Optimistic:** Higher redemption (55%), higher margins, higher adoption
            
            *Use these ranges to plan budgets and set realistic expectations.*
            """)
    
   
    # Sources
   
    elif menu == "ℹ️ Sources":
        st.title("ℹ️ Scientific sources & methodology")
        
        if "metadata" in data:
            meta = data["metadata"]
            
            st.subheader("📚 Scientific sources & quotes")
            if "scientific_sources" in meta:
                for source in meta["scientific_sources"]:
                    st.subheader(source["source"])
                    if "doi" in source:
                        st.caption(f"DOI: {source['doi']}")
                    elif "url" in source:
                        st.caption(f"URL: {source['url']}")
                    
                    for quote in source["quotes"]:
                        st.markdown(f"> *{quote}*")
                    
                    st.caption(f"Used for: {source['used_for']}")
                    st.divider()
            
            st.subheader("Department margin sources")
            if "margins" in meta and "sources" in meta["margins"]:
                margin_sources = meta["margins"]["sources"]
                for dept, info in margin_sources.items():
                    st.markdown(f"**{dept.title()} ({info['value']*100:.0f}% margin)**")
                    st.caption(f"Source: {info['source']}")
                    st.caption(f"Quote: *{info['quote']}*")
                    st.divider()
            
            st.warning("""
            ⚠️ **Important notes**:
            1. **Gross vs. net margins:** All margin figures are gross margins (before operating expenses). Net profitability depends on your store's specific cost structure.
            2. **Projections, not guarantees:** These are research-backed estimates. Actual results may vary.
            3. **Test before scaling:** Start with a small A/B test before rolling out to all customers.
            
            *For profit calculations, apply your internal cost-of-goods data to the revenue figures above.*
            """)

# Run app
if __name__ == "__main__":
    main()