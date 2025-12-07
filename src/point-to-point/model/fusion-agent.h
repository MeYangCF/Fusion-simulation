#ifndef FUSION_AGENT_H
#define FUSION_AGENT_H

#include <ns3/object.h>
#include <ns3/data-rate.h>
#include <ns3/time.h>
#include <ns3/rdma-queue-pair.h>
#include <vector>
#include <map>
#include <ns3/event-id.h>

namespace ns3 {

// Forward declaration
class FusionGymEnv;

class FusionAgent : public Object {
public:
	static TypeId GetTypeId (void);
	FusionAgent();
	virtual ~FusionAgent();

	struct NetworkState {
		double thr;          // average throughput
		double rtt;          // average packet delay
		double thr_max;      // maximum throughput observed
		double rtt_min;      // minimum packet delay observed
		double loss;         // average loss rate
		double delta_Rt;     // ratio change in delivery rate
		double Occ;          // average switch buffer occupancy rate
		double prev_target_rate; // previous target rate (for delta_Rt calculation)
		
		NetworkState() : thr(0), rtt(0), thr_max(0), rtt_min(1e9), 
		                 loss(0), delta_Rt(0), Occ(0), prev_target_rate(0) {}
	};

	struct FusionParams {
		double delta;  // 0.5 <= delta <= 1.5
		double gamma;  // 0.5 <= gamma <= 1.5
		double beta;   // 0 < beta < 1
		double mu1;    // 2 < mu1 <= mu2 <= 4
		double mu2;    // 2 < mu1 <= mu2 <= 4
		double mu3;    // 6 <= mu3 <= 10
		
		FusionParams() : delta(1.0), gamma(1.0), beta(0.1), 
		                 mu1(2.5), mu2(3.0), mu3(8.0) {}
	};

	void Initialize();
	void CollectStatistics(Ptr<RdmaQueuePair> qp, NetworkState &state);
	FusionParams GenerateParameters(const NetworkState &state);
	void SetUseGym(bool useGym);
	bool GetUseGym() const;
	void SetGymEnv(Ptr<FusionGymEnv> env);
	Ptr<FusionGymEnv> GetGymEnv() const;
	void StartMonitoring(Ptr<RdmaQueuePair> qp);
	double CalculateReward(const NetworkState &state);
	void UpdateAgent(const NetworkState &state, double reward);
	void SetMonitoringInterval(Time interval);
	Time GetMonitoringInterval() const;

private:
	Time m_monitoringInterval;
	double m_omega1;
	double m_omega2;
	double m_omega3;
	double m_omega4;
	std::map<Ptr<RdmaQueuePair>, NetworkState> m_qpStates;
	struct QpParamState {
		FusionParams lastParams;
		Time lastUpdateTime;
		bool paramsPending;
		Time pendingSince;
		QpParamState() : lastUpdateTime(Time(0)), paramsPending(false), pendingSince(Time(0)) {}
	};
	std::map<Ptr<RdmaQueuePair>, QpParamState> m_qpParamStates;
	FusionParams m_defaultParams;
	bool m_useGym;
	Ptr<FusionGymEnv> m_gymEnv;
	std::map<Ptr<RdmaQueuePair>, EventId> m_qpMonitoringEvents;
	Time m_paramUpdateTimeout;
	void NormalizeState(NetworkState &state);
	FusionParams AdjustParameters(const NetworkState &state);
	void ScheduleNextMonitoring(Ptr<RdmaQueuePair> qp);
	void MonitoringIntervalCallback(Ptr<RdmaQueuePair> qp);
	void CheckPendingParameters(Ptr<RdmaQueuePair> qp);
	void ApplyParameters(Ptr<RdmaQueuePair> qp, const FusionParams &params);
	void MarkParametersUpdated(Ptr<RdmaQueuePair> qp, const FusionParams &params);
};

} /* namespace ns3 */

#endif /* FUSION_AGENT_H */

