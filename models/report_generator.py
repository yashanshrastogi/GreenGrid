import os
import tempfile
from fpdf import FPDF
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime

def format_whatsapp_alert(row, data_source):
    """Simple text formatter for WhatsApp alerts."""
    loc = row.get("location", "?")
    if "meter_id" in row and pd.isna(loc):
        loc = row.get("meter_id")
    ts = str(row.get("timestamp", ""))[:16]
    pw = row.get("power_kw", 0.0)
    ex = row.get("expected_load", 0.0)
    dev = row.get("deviation_index", 0.0)
    act = row.get("action", "")
    w = row.get("waste_kwh", 0.0)
    c = row.get("cost_saved_inr", 0.0)
    co2 = row.get("co2e_avoided_kg", 0.0)
    
    return f"""⚡ *GreenGrid Energy Alert*
📊 Data Source: {data_source}

📍 Location : {loc}
🕒 Time     : {ts}
⚡ Load     : {pw:.3f} kW  (expected: {ex:.3f} kW)
🔍 Dev.Score: {dev:.3f}
▶️ Action   : {act}

💡 Est. Waste: {w:.4f} kWh
💰 Cost Impact: ₹{c:.2f}
🌱 CO₂e Avoided: {co2:.4f} kg

GreenGrid • Energy Intelligence System"""

def generate_pdf_report(df, data_source, top_n=10, period_label="Period"):
    """Generate a comprehensive PDF report using FPDF."""
    class PDF(FPDF):
        def header(self):
            self.set_font("helvetica", "B", 15)
            self.cell(0, 10, "GreenGrid - Energy Intelligence System", border=False, ln=1, align="C")
            self.set_font("helvetica", "I", 10)
            self.cell(0, 10, f"Comprehensive Energy Report | Source: {data_source} | {period_label}", border=False, ln=1, align="C")
            self.ln(5)
            
        def footer(self):
            self.set_y(-15)
            self.set_font("helvetica", "I", 8)
            self.cell(0, 10, f"Page {self.page_no()} | Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}", border=False, align="C")
            
    pdf = PDF(orientation="P", unit="mm", format="A4")
    pdf.add_page()
    
    # 1. Executive Summary
    pdf.set_font("helvetica", "B", 14)
    pdf.cell(0, 10, "1. Executive Summary", ln=1)
    
    df_f = df[df["action"] != "normal"]
    tot_w = df["waste_kwh"].sum()
    tot_c = df["cost_saved_inr"].sum()
    tot_co2 = df["co2e_avoided_kg"].sum()
    
    pdf.set_font("helvetica", "", 11)
    pdf.cell(0, 8, f"Total Records Analyzed: {len(df):,}", ln=1)
    pdf.cell(0, 8, f"Flagged Events: {len(df_f):,} ({(len(df_f)/max(len(df),1)*100):.1f}%)", ln=1)
    pdf.cell(0, 8, f"Energy Waste Caught: {tot_w:.2f} kWh", ln=1)
    pdf.cell(0, 8, f"Estimated Cost Savings: Rs. {tot_c:,.2f}", ln=1)
    pdf.cell(0, 8, f"CO2e Avoided: {tot_co2:.2f} kg", ln=1)
    pdf.ln(5)
    
    # 2. Charts
    pdf.set_font("helvetica", "B", 14)
    pdf.cell(0, 10, "2. Analytics & Visualizations", ln=1)
    
    temp_dir = tempfile.mkdtemp()
    
    # Action Distribution Chart
    ad_file = os.path.join(temp_dir, "action_dist.png")
    plt.figure(figsize=(8, 4))
    action_counts = df["action"].value_counts()
    colors = ["#34d399" if act=="normal" else "#f87171" if act=="critical-flag" else "#fbbf24" for act in action_counts.index]
    action_counts.plot(kind="bar", color=colors)
    plt.title("Action Distribution")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(ad_file, dpi=150)
    plt.close()
    
    # Deviation Histogram
    hist_file = os.path.join(temp_dir, "deviation_hist.png")
    plt.figure(figsize=(8, 4))
    plt.hist(df["deviation_index"].dropna(), bins=50, color="#818cf8", edgecolor="black")
    plt.title("Deviation Index Distribution")
    plt.xlabel("Deviation Score")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(hist_file, dpi=150)
    plt.close()
    
    # Time series of power
    ts_file = os.path.join(temp_dir, "timeseries.png")
    plt.figure(figsize=(8, 4))
    if len(df) > 1000:
        sample_df = df.sample(1000).sort_values("timestamp")
    else:
        sample_df = df.sort_values("timestamp")
    plt.plot(sample_df["timestamp"], sample_df["power_kw"], color="#00d4aa", alpha=0.7)
    plt.title("Power Consumption Over Time (Sampled)")
    plt.ylabel("kW")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(ts_file, dpi=150)
    plt.close()
    
    # Place images nicely
    curr_y = pdf.get_y()
    pdf.image(ad_file, x=10, y=curr_y, w=90)
    pdf.image(hist_file, x=110, y=curr_y, w=90)
    pdf.set_y(curr_y + 50)
    pdf.image(ts_file, x=15, y=pdf.get_y(), w=180)
    pdf.set_y(pdf.get_y() + 95)
    
    # 3. Insights
    pdf.add_page()
    pdf.set_font("helvetica", "B", 14)
    pdf.cell(0, 10, "3. Key Findings & Recommendations", ln=1)
    
    flag_rate = len(df_f) / max(len(df), 1) * 100
    industry_avg = 3.5
    
    pdf.set_font("helvetica", "B", 12)
    pdf.cell(0, 8, "Anomaly Rate:", ln=1)
    pdf.set_font("helvetica", "", 11)
    if flag_rate > industry_avg * 1.5:
        pdf.multi_cell(0, 6, f"Your dataset shows a {flag_rate:.1f}% event rate — significantly above the {industry_avg}% industry benchmark. This suggests systematic over-consumption or faulty sensor readings. Priority investigation recommended.")
    elif flag_rate > industry_avg:
        pdf.multi_cell(0, 6, f"{flag_rate:.1f}% of readings flagged — slightly above the {industry_avg}% benchmark. Review the top flagged locations for scheduling mismatches.")
    else:
        pdf.multi_cell(0, 6, f"{flag_rate:.1f}% event rate is within the {industry_avg}% benchmark. Energy consumption patterns appear well-controlled.")
    pdf.ln(4)
    
    location_col = "location" if "location" in df.columns else "meter_id"
    if location_col in df.columns and not df_f.empty:
        top_loc = df_f.groupby(location_col)["waste_kwh"].sum().idxmax()
        top_waste = df_f.groupby(location_col)["waste_kwh"].sum().max()
        pdf.set_font("helvetica", "B", 12)
        pdf.cell(0, 8, f"Highest Waste Location: {top_loc}", ln=1)
        pdf.set_font("helvetica", "", 11)
        pdf.multi_cell(0, 6, f"'{top_loc}' accounts for {top_waste:.2f} kWh of recovered waste energy in the selected period. Consider scheduling an audit of equipment in this area.")
        pdf.ln(4)
        
    filter_days = (df["timestamp"].max() - df["timestamp"].min()).days
    if filter_days == 0: filter_days = 1
    
    # Peak Hour Insight
    df["hour"] = df["timestamp"].dt.hour
    peak_hour = df.groupby("hour")["deviation_index"].mean().idxmax()
    pdf.set_font("helvetica", "B", 12)
    pdf.cell(0, 8, f"Peak Deviation Hour: {peak_hour:02d}:00", ln=1)
    pdf.set_font("helvetica", "", 11)
    pdf.multi_cell(0, 6, f"Anomaly scores are systematically highest at {peak_hour:02d}:00. If this does not match an expected high-use period, investigate equipment schedules or HVAC behaviour.")
    pdf.ln(4)
    
    # Cost Risk
    proj_annual = tot_c * (365 / filter_days)
    if proj_annual > 1000:
        pdf.set_font("helvetica", "B", 12)
        pdf.cell(0, 8, "Projected Annual Cost Risk:", ln=1)
        pdf.set_font("helvetica", "", 11)
        pdf.multi_cell(0, 6, f"Based on the current window, projected annual energy waste could cost Rs. {proj_annual:,.0f}. Automated scheduling policies or occupancy-based controls could recover 40-60% of this.")
        pdf.ln(4)
        
    # Carbon Footprint
    tot_co2 = df["co2e_avoided_kg"].sum()
    proj_co2 = tot_co2 * (365 / filter_days)
    pdf.set_font("helvetica", "B", 12)
    pdf.cell(0, 8, "Carbon Footprint Projection:", ln=1)
    pdf.set_font("helvetica", "", 11)
    pdf.multi_cell(0, 6, f"Projected annual CO2 avoidance at current performance: {proj_co2:.1f} kg CO2e (equivalent to planting ~{int(proj_co2/21)} trees). Full occupancy-based automation could double this figure.")
    pdf.ln(6)
    
    # 4. Recommendations
    pdf.set_font("helvetica", "B", 14)
    pdf.cell(0, 10, "4. Strategic Recommendations", ln=1)
    
    recs = [
        ("Implement Occupancy-Based Controls", "Install PIR or Wi-Fi presence sensors and integrate with your energy management system. This alone reduces Critical Flag events by 30-50% in unoccupied spaces."),
        ("Set Automated Scheduling Policies", "Program HVAC and lighting to follow room booking calendars. This eliminates the most common cause of after-hours Waste spikes."),
        ("Review Peak-Hour Equipment", f"Focus maintenance on equipment active during the peak deviation hour ({peak_hour:02d}:00). Aging HVAC compressors and water heaters are common culprits."),
        ("Establish Baseline Monitoring", "Re-run this analysis monthly to track whether interventions reduce the flagged event rate. Target: below 3.5%."),
        ("Upgrade to Sub-Metering", "Break down energy tracking to individual circuits within each room. Sub-metering isolates the exact appliance causing High Watch events and reduces investigation time by over 60%."),
    ]
    
    pdf.set_font("helvetica", "", 11)
    for title, body in recs:
        pdf.set_font("helvetica", "B", 11)
        pdf.cell(0, 6, f"- {title}:", ln=1)
        pdf.set_font("helvetica", "", 11)
        pdf.set_x(15)
        pdf.multi_cell(0, 6, body)
        pdf.ln(2)
        
    pdf.add_page()
    
    # 5. Top Alerts Table
    pdf.set_font("helvetica", "B", 14)
    pdf.cell(0, 10, f"5. Top {top_n} Anomalies by Deviation Index", ln=1)
    
    if df_f.empty:
        pdf.set_font("helvetica", "I", 11)
        pdf.cell(0, 8, "No anomalous events found in the selected period.", ln=1)
    else:
        top_events = df_f.sort_values("deviation_index", ascending=False).head(top_n)
        
        # Table Header
        pdf.set_font("helvetica", "B", 10)
        pdf.cell(40, 8, "Timestamp", 1)
        pdf.cell(50, 8, "Location/Meter", 1)
        pdf.cell(25, 8, "Power (kW)", 1)
        pdf.cell(25, 8, "Deviation", 1)
        pdf.cell(30, 8, "Waste (kWh)", 1)
        pdf.cell(20, 8, "Action", 1, ln=1)
        
        # Table Rows
        pdf.set_font("helvetica", "", 9)
        for _, row in top_events.iterrows():
            loc = str(row.get(location_col, "?"))[:20]
            ts = str(row["timestamp"])[:16]
            pw = f"{row.get('power_kw', 0):.2f}"
            dev = f"{row.get('deviation_index', 0):.2f}"
            wst = f"{row.get('waste_kwh', 0):.3f}"
            act = str(row.get("action", ""))
            
            pdf.cell(40, 7, ts, 1)
            pdf.cell(50, 7, loc, 1)
            pdf.cell(25, 7, pw, 1)
            pdf.cell(25, 7, dev, 1)
            pdf.cell(30, 7, wst, 1)
            pdf.cell(20, 7, act, 1, ln=1)
            
    # Cleanup temps
    for f in [ad_file, hist_file, ts_file]:
        try: os.remove(f)
        except: pass
    try: os.rmdir(temp_dir)
    except: pass
    
    # Return as bytes for Streamlit compatibility
    return bytes(pdf.output())
