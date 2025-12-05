### **How to Interpret the Output**

The script will print a summary table after each 60-second stage. Here's what to look for:

1.  **Early Stages (Low Concurrency):**
    *   **Failure Rate:** Should be `0.00%`.
    *   **Latencies:** Should be relatively low and stable.
    *   **AWS Console:** Your 2 instances will be handling the load, and their CPU utilization will be rising.

2.  **Middle Stages (Approaching the Limit):**
    *   **Latencies:** You'll see the Average and especially the 95th Percentile Latency start to increase significantly. This is the first sign of strain.
    *   **AWS Console:** The average CPU will cross your threshold. The CloudWatch alarm will trigger, and the ASG will start launching a **new instance**.

3.  **Breaking Point (High Concurrency):**
    *   **Failure Rate:** Will jump from 0% to a higher number.
    *   **Failure Breakdown:** This is the most important part. You will see errors like:
        *   **`timeout`**: Your most likely first failure. Your application is so busy it can't respond within the 30-second window.
        *   **`503 ServiceUnavailable`**: The ALB might report this if all its targets are overwhelmed.
        *   **`502 BadGateway`**: This can happen if an instance crashes or the application service inside the container stops responding.
    *   The script will automatically stop when the failure rate exceeds 5%. The last successful concurrency level is the approximate capacity of your fleet *before* it scales.

By running this test, you will get a clear, data-driven answer to "at what speed do requests start to fail" and you will get to see your autoscaling system kick in and save the day in real-time.
