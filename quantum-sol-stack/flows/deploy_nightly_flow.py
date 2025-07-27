"""
Deployment configuration for the QuantumSol Nightly Flow.

This script creates a Prefect deployment for the nightly maintenance and optimization flow.
"""
from prefect.deployments import Deployment
from prefect.orion.schemas.schedules import IntervalSchedule
from datetime import datetime, timedelta, timezone, time as time_type
import pendulum

from nightly_flow import nightly_flow

# Define the schedule to run daily at 1:00 AM UTC (after market close)
schedule = IntervalSchedule(
    interval=timedelta(days=1),
    timezone="UTC",
    anchor_date=pendulum.datetime(2023, 1, 1, 1, 0, tz="UTC"),  # First run at 1:00 AM UTC
)

# Create the deployment
deployment = Deployment.build_from_flow(
    flow=nightly_flow,
    name="quantumsol-nightly-production",
    version="1.0.0",
    schedule=schedule,
    parameters={
        "test_mode": False
    },
    tags=["production", "nightly", "maintenance"],
    description="Nightly maintenance and optimization flow for QuantumSol trading system",
    work_queue_name="production"
)

if __name__ == "__main__":
    # Create the deployment in the Prefect database
    deployment.apply()
    print("Successfully created/updated deployment for QuantumSol Nightly Flow")
