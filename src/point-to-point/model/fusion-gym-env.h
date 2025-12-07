#ifndef FUSION_GYM_ENV_H
#define FUSION_GYM_ENV_H

#ifdef NS3_OPENGYM
#include <ns3/opengym-interface.h>
#endif
#include <ns3/rdma-queue-pair.h>
#include "fusion-agent.h"
#include <vector>

namespace ns3 {

#ifdef NS3_OPENGYM
class FusionGymEnv : public OpenGymInterface {
#else
class FusionGymEnv : public Object {
#endif
public:
	static TypeId GetTypeId (void);
	FusionGymEnv();
	virtual ~FusionGymEnv();
	void SetFusionAgent(Ptr<FusionAgent> agent);
	Ptr<FusionAgent> GetFusionAgent() const;
	void SetQueuePair(Ptr<RdmaQueuePair> qp);
	Ptr<RdmaQueuePair> GetQueuePair() const;
	void NotifyMonitoringInterval(Ptr<RdmaQueuePair> qp, const FusionAgent::NetworkState &state);
	void SetPort(uint32_t port);
	uint32_t GetPort() const;

protected:
#ifdef NS3_OPENGYM
	virtual Ptr<OpenGymSpace> GetObservationSpace();
	virtual Ptr<OpenGymSpace> GetActionSpace();
	virtual Ptr<OpenGymDataContainer> GetObservation();
	virtual Ptr<OpenGymDataContainer> GetReward();
	virtual bool GetGameOver();
	virtual std::string GetExtraInfo();
	virtual bool ExecuteActions(Ptr<OpenGymDataContainer> action);
#endif

private:
	Ptr<FusionAgent> m_agent;
	Ptr<RdmaQueuePair> m_qp;
	uint32_t m_port;
	FusionAgent::NetworkState m_currentState;
	FusionAgent::NetworkState m_previousState;
	double m_currentReward;
	bool m_gameOver;
	double m_episodeReward;
	uint32_t m_episodeStep;
	std::map<Ptr<RdmaQueuePair>, FusionAgent::NetworkState> m_qpStates;
	std::map<Ptr<RdmaQueuePair>, FusionAgent::FusionParams> m_qpParams;
	std::vector<double> StateToObservation(const FusionAgent::NetworkState &state);
#ifdef NS3_OPENGYM
	FusionAgent::FusionParams ActionToParams(Ptr<OpenGymDataContainer> action);
#endif
};

} /* namespace ns3 */

#endif /* FUSION_GYM_ENV_H */

