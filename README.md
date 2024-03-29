# Capacity-Aware Fair POI Recommendation Combining Transformer Neural Networks and Resource Allocation Policy (CAFPR)

Point of Interest (POI) recommendations have focused on maximising user satisfaction but have mostly ignored the needs of POIs and their operators. One such need is recommendation exposure.  This can result in envy among the POIs, some of whom are under recommended with respect to their capacity, others who are over which can cause dissatisfication for both staff at the POI and users who might have to queue long time for entry or experience overcrowding. Existing work has not addressed this trade-off between satisfying user preferences and being fair to POIs, which typically seek to be at capacity. Therefore we introduce the POI fair allocation problem to model this problem, considering user satisfaction and POI exposure fairness. To address this problem, we propose a fair POI allocation technique that balances user satisfaction and POI capacity-based exposure simultaneously. The proposed model uses existing personalised POI recommendation model that captures users to POI visits' spatio-temporal influences and user interests. After that, we propose POI capacity-based allocation using the over-demand cut policy and under-demand add policy. This allocation ensures POIs exposure ratio and envy-free up to certain thresholds. We evaluate our proposed model performance on five datasets containing real-life POI visits. The experimental evaluations show that the proposed model outperforms the baselines in terms of user and POI-based evaluation metrics.

To use this code in your research work please cite the following paper.

Sajal Halder, Kwan Hui Lim, Jeﬀrey Chan, and Xiuzhen Zhang. Capacity-Aware Fair POI Recommendation Combining Transformer Neural Networks and Resource Allocation Policy. Submitted to Applied Soft Computing,  2023.

# Implemtation Details
In this CAFPR model implemenation, we have used existing recommendation model and our proposed capacity aware over demand and underdemand adjust policy. 

Required Packages:

tensorflow: 2.4.1
pandas: 1.2.0

CAFPR model has been implemented in ATLSTM_CAFPR_mail.py file.

Here we added only one dataset (California Advencture). If you are interested to know about more datasets email at sajal.csedu01@gmail.com
