"""
SwasthStock Complete Demo
=========================

This script demonstrates the full SwasthStock AI pipeline:
1. Load historical medicine consumption data
2. Train demand forecasting model
3. Predict next 90 days consumption
4. Detect expiry risks
5. Generate reorder recommendations
6. Calculate waste savings

Run: python run_demo.py
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from demand_forecast import (
    MedicineDemandForecaster, 
    detect_expiry_risks, 
    calculate_waste_savings
)


def generate_synthetic_data(num_medicines=20, num_days=180):
    """
    Generate synthetic hospital medicine consumption data for demo.
    
    In production, this would be replaced with real hospital data from:
    - Hospital ERP/HIS system
    - Pharmacy management software
    - Purchase order records
    """
    print("🏥 Generating synthetic hospital consumption data...")
    
    # Common Indian hospital medicines
    medicines = [
        {'name': 'Paracetamol 500mg', 'category': 'Analgesic', 'base_daily': 120, 'seasonality': 'winter'},
        {'name': 'Amoxicillin 250mg', 'category': 'Antibiotic', 'base_daily': 80, 'seasonality': 'monsoon'},
        {'name': 'Metformin 500mg', 'category': 'Antidiabetic', 'base_daily': 150, 'seasonality': 'none'},
        {'name': 'Amlodipine 5mg', 'category': 'Antihypertensive', 'base_daily': 100, 'seasonality': 'none'},
        {'name': 'Azithromycin 500mg', 'category': 'Antibiotic', 'base_daily': 60, 'seasonality': 'monsoon'},
        {'name': 'Cetirizine 10mg', 'category': 'Antihistamine', 'base_daily': 90, 'seasonality': 'summer'},
        {'name': 'Omeprazole 20mg', 'category': 'Antacid', 'base_daily': 110, 'seasonality': 'none'},
        {'name': 'Atorvastatin 10mg', 'category': 'Statin', 'base_daily': 85, 'seasonality': 'none'},
        {'name': 'Insulin Glargine', 'category': 'Antidiabetic', 'base_daily': 45, 'seasonality': 'none'},
        {'name': 'Aspirin 75mg', 'category': 'Antiplatelet', 'base_daily': 95, 'seasonality': 'none'},
        {'name': 'Ciprofloxacin 500mg', 'category': 'Antibiotic', 'base_daily': 70, 'seasonality': 'summer'},
        {'name': 'Dexamethasone 4mg', 'category': 'Steroid', 'base_daily': 50, 'seasonality': 'none'},
        {'name': 'Diazepam 5mg', 'category': 'Anxiolytic', 'base_daily': 30, 'seasonality': 'none'},
        {'name': 'Ibuprofen 400mg', 'category': 'NSAID', 'base_daily': 100, 'seasonality': 'winter'},
        {'name': 'Ranitidine 150mg', 'category': 'Antacid', 'base_daily': 75, 'seasonality': 'none'},
        {'name': 'Salbutamol Inhaler', 'category': 'Bronchodilator', 'base_daily': 40, 'seasonality': 'winter'},
        {'name': 'Chloroquine 250mg', 'category': 'Antimalarial', 'base_daily': 25, 'seasonality': 'monsoon'},
        {'name': 'Doxycycline 100mg', 'category': 'Antibiotic', 'base_daily': 55, 'seasonality': 'monsoon'},
        {'name': 'Furosemide 40mg', 'category': 'Diuretic', 'base_daily': 65, 'seasonality': 'none'},
        {'name': 'Levothyroxine 50mcg', 'category': 'Thyroid', 'base_daily': 80, 'seasonality': 'none'}
    ]
    
    # Generate daily consumption records
    records = []
    start_date = datetime.now() - timedelta(days=num_days)
    
    for medicine in medicines[:num_medicines]:
        for day in range(num_days):
            date = start_date + timedelta(days=day)
            month = date.month
            
            # Base consumption
            consumption = medicine['base_daily']
            
            # Add seasonality
            if medicine['seasonality'] == 'winter' and month in [12, 1, 2]:
                consumption *= 1.5  # 50% increase in winter
            elif medicine['seasonality'] == 'monsoon' and month in [6, 7, 8, 9]:
                consumption *= 1.8  # 80% increase in monsoon (infections)
            elif medicine['seasonality'] == 'summer' and month in [3, 4, 5]:
                consumption *= 1.3  # 30% increase in summer
            
            # Add weekly pattern (lower on weekends)
            if date.weekday() in [5, 6]:
                consumption *= 0.7
            
            # Add random variation
            consumption = consumption * np.random.uniform(0.85, 1.15)
            
            # Add hospital bed occupancy effect
            hospital_beds_occupied = np.random.randint(200, 300)
            
            records.append({
                'date': date,
                'medicine_name': medicine['name'],
                'category': medicine['category'],
                'daily_consumption': round(consumption, 2),
                'hospital_beds_occupied': hospital_beds_occupied
            })
    
    df = pd.DataFrame(records)
    print(f"✅ Generated {len(df)} consumption records for {num_medicines} medicines over {num_days} days\n")
    return df


def generate_current_inventory(medicines, forecast_df):
    """
    Generate current inventory with expiry dates for demo.
    """
    inventory = []
    
    for medicine in medicines:
        # Random current stock (30-120 days worth)
        avg_daily = forecast_df[forecast_df['medicine_name'] == medicine]['daily_consumption'].mean()
        current_stock = int(avg_daily * np.random.uniform(30, 120))
        
        # Random expiry date (30-365 days from now)
        days_to_expiry = np.random.randint(30, 365)
        expiry_date = datetime.now() + timedelta(days=days_to_expiry)
        
        # Random unit price
        unit_price = np.random.uniform(2, 50)
        
        inventory.append({
            'medicine_name': medicine,
            'current_stock': current_stock,
            'expiry_date': expiry_date,
            'unit_price_rs': unit_price
        })
    
    return pd.DataFrame(inventory)


def main():
    """
    Run complete SwasthStock demo.
    """
    print("\n" + "="*70)
    print("         SWASTHSTOCK AI - COMPLETE DEMO")
    print("="*70 + "\n")
    
    # Step 1: Generate data
    print("STEP 1: Load Historical Consumption Data")
    print("-" * 70)
    consumption_data = generate_synthetic_data(num_medicines=20, num_days=180)
    
    print("Sample data:")
    print(consumption_data.head(10))
    print(f"\nDate range: {consumption_data['date'].min()} to {consumption_data['date'].max()}")
    print(f"Total medicines tracked: {consumption_data['medicine_name'].nunique()}")
    print(f"Total records: {len(consumption_data)}\n")
    
    # Step 2: Train model
    print("\nSTEP 2: Train AI Demand Forecasting Model")
    print("-" * 70)
    forecaster = MedicineDemandForecaster(model_type='random_forest')
    metrics = forecaster.train(consumption_data, target_column='daily_consumption')
    
    # Step 3: Predict future demand
    print("\n\nSTEP 3: Predict Demand for Next 90 Days")
    print("-" * 70)
    predictions = forecaster.predict_demand(consumption_data, days_ahead=90)
    
    print(f"✅ Generated {len(predictions)} predictions for next 90 days\n")
    print("Sample predictions for Paracetamol:")
    paracetamol_pred = predictions[predictions['medicine_name'] == 'Paracetamol 500mg'].head(10)
    print(paracetamol_pred[['date', 'predicted_daily_consumption', 'predicted_monthly_consumption']])
    
    # Step 4: Generate current inventory
    print("\n\nSTEP 4: Analyze Current Inventory")
    print("-" * 70)
    medicines = consumption_data['medicine_name'].unique()
    inventory = generate_current_inventory(medicines, consumption_data)
    
    print("Current inventory snapshot:")
    print(inventory.head(10))
    
    # Step 5: Detect expiry risks
    print("\n\nSTEP 5: Detect Medicines at Risk of Expiry")
    print("-" * 70)
    expiry_alerts = detect_expiry_risks(inventory, alert_days=[30, 60, 90])
    
    if len(expiry_alerts) > 0:
        print(f"⚠️  Found {len(expiry_alerts)} medicines at risk of expiry:\n")
        print(expiry_alerts.to_string(index=False))
        
        total_risk_value = expiry_alerts['potential_waste_value_rs'].sum()
        print(f"\n💰 Total potential waste value: ₹{total_risk_value:,.2f}")
    else:
        print("✅ No medicines at immediate risk of expiry")
    
    # Step 6: Generate reorder recommendations
    print("\n\nSTEP 6: Generate Smart Reorder Recommendations")
    print("-" * 70)
    recommendations = forecaster.generate_reorder_recommendations(
        current_stock=inventory,
        predictions=predictions,
        lead_time_days=7,
        safety_stock_factor=1.2
    )
    
    print("🎯 Reorder Recommendations (sorted by urgency):\n")
    print(recommendations.head(10).to_string(index=False))
    
    urgent_reorders = recommendations[recommendations['reorder_now'] == True]
    print(f"\n🔴 Urgent reorders needed: {len(urgent_reorders)} medicines")
    print(f"🟡 Total medicines monitored: {len(recommendations)}")
    
    # Step 7: Calculate savings
    print("\n\nSTEP 7: Calculate Waste Reduction & Savings")
    print("-" * 70)
    
    # Typical district hospital medicine budget
    savings = calculate_waste_savings(
        current_waste_percent=18,      # Industry average in India
        target_waste_percent=5,        # With SwasthStock AI
        annual_medicine_budget_rs=1_00_00_000  # ₹1 Crore annual budget
    )
    
    # Summary report
    print("\n\n" + "="*70)
    print("                    DEMO SUMMARY REPORT")
    print("="*70)
    print(f"✅ Model Accuracy: {metrics['r2_score']*100:.2f}%")
    print(f"✅ Medicines Tracked: {len(medicines)}")
    print(f"✅ Predictions Generated: {len(predictions)} (90 days ahead)")
    print(f"⚠️  Expiry Risks Detected: {len(expiry_alerts)}")
    print(f"🎯 Reorder Recommendations: {len(recommendations)}")
    print(f"💰 Annual Savings Potential: ₹{savings['annual_savings_rs']:,.0f}")
    print(f"📊 Waste Reduction: {savings['waste_reduction_percent']:.1f}%")
    print("="*70)
    
    print("\n\n🎉 DEMO COMPLETE!")
    print("\nNext Steps:")
    print("1. Deploy in pilot hospital to collect real data")
    print("2. Integrate with hospital ERP/HIS system")
    print("3. Launch web dashboard for procurement officers")
    print("4. Scale to 25 hospitals across UP")
    print("\nFor questions: Contact SwasthStock team")
    print("="*70 + "\n")
    
    # Save outputs
    print("💾 Saving outputs...")
    predictions.to_csv('data/predictions_next_90_days.csv', index=False)
    recommendations.to_csv('data/reorder_recommendations.csv', index=False)
    if len(expiry_alerts) > 0:
        expiry_alerts.to_csv('data/expiry_alerts.csv', index=False)
    
    print("✅ Saved to data/ folder:")
    print("   - predictions_next_90_days.csv")
    print("   - reorder_recommendations.csv")
    print("   - expiry_alerts.csv")
    print("\n")


if __name__ == "__main__":
    main()
