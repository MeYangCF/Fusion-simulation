#include "fusion-gym-env.h"
#include <ns3/log.h>
#include <ns3/simulator.h>
#include <algorithm>
#include <cmath>

#ifdef NS3_OPENGYM
#include <ns3/opengym-interface.h>
#endif

NS_LOG_COMPONENT_DEFINE("FusionGymEnv");

namespace ns3 {

NS_OBJECT_ENSURE_REGISTERED(FusionGymEnv);

TypeId FusionGymEnv::GetTypeId (void)
{
#ifdef NS3_OPENGYM
	static TypeId tid = TypeId ("ns3::FusionGymEnv")
		.SetParent<OpenGymInterface> ()
		.AddConstructor<FusionGymEnv> ()
		.AddAttribute("Port",
				"Port number for gym communication",
				UintegerValue(5555),
				MakeUintegerAccessor(&FusionGymEnv::m_port),
				MakeUintegerChecker<uint32_t>())
		;
	return tid;
#else
	static TypeId tid = TypeId ("ns3::FusionGymEnv")
		.SetParent<Object> ()
		.AddConstructor<FusionGymEnv> ()
		.AddAttribute("Port",
				"Port number for gym communication",
				UintegerValue(5555),
				MakeUintegerAccessor(&FusionGymEnv::m_port),
				MakeUintegerChecker<uint32_t>())
		;
	return tid;
#endif
}

FusionGymEnv::FusionGymEnv()
	: m_agent(0),
	  m_qp(0),
	  m_currentReward(0.0),
	  m_gameOver(false),
	  m_episodeReward(0.0),
	  m_episodeStep(0),
	  m_port(5555)
{
	NS_LOG_FUNCTION(this);
}

FusionGymEnv::~FusionGymEnv()
{
	NS_LOG_FUNCTION(this);
}

void FusionGymEnv::SetFusionAgent(Ptr<FusionAgent> agent)
{
	NS_LOG_FUNCTION(this << agent);
	m_agent = agent;
}

Ptr<FusionAgent> FusionGymEnv::GetFusionAgent() const
{
	NS_LOG_FUNCTION(this);
	return m_agent;
}

void FusionGymEnv::SetQueuePair(Ptr<RdmaQueuePair> qp)
{
	NS_LOG_FUNCTION(this << qp);
	m_qp = qp;
}

Ptr<RdmaQueuePair> FusionGymEnv::GetQueuePair() const
{
	NS_LOG_FUNCTION(this);
	return m_qp;
}

void FusionGymEnv::NotifyMonitoringInterval(Ptr<RdmaQueuePair> qp, const FusionAgent::NetworkState &state)
{
	NS_LOG_FUNCTION(this << qp);
	if (!m_agent || !qp) {
		return;
	}
	auto prevIt = m_qpStates.find(qp);
	if (prevIt != m_qpStates.end()) {
		m_previousState = prevIt->second;
	} else {
		m_previousState = FusionAgent::NetworkState();
	}
	m_currentState = state;
	m_qpStates[qp] = state;
	m_qp = qp;
	m_currentReward = m_agent->CalculateReward(m_currentState);
	m_episodeReward += m_currentReward;
	m_episodeStep++;
#ifdef NS3_OPENGYM
	Notify();
#endif
}

void FusionGymEnv::SetPort(uint32_t port)
{
	NS_LOG_FUNCTION(this << port);
	m_port = port;
}

uint32_t FusionGymEnv::GetPort() const
{
	NS_LOG_FUNCTION(this);
	return m_port;
}

#ifdef NS3_OPENGYM
Ptr<OpenGymSpace> FusionGymEnv::GetObservationSpace()
{
	NS_LOG_FUNCTION(this);
	uint32_t nObs = 7;
	Ptr<OpenGymBoxSpace> space = CreateObject<OpenGymBoxSpace> ();
	std::vector<double> low(nObs, 0.0);
	std::vector<double> high(nObs, 1.0);
	high[2] = 10.0;
	high[3] = 10.0;
	high[4] = 1.0;
	high[5] = 2.0;
	high[6] = 1.0;
	space->SetLow(low);
	space->SetHigh(high);
	return space;
}

Ptr<OpenGymSpace> FusionGymEnv::GetActionSpace()
{
	NS_LOG_FUNCTION(this);
	uint32_t nAct = 6;
	Ptr<OpenGymBoxSpace> space = CreateObject<OpenGymBoxSpace> ();
	std::vector<double> low = {0.5, 0.5, 0.01, 2.1, 2.1, 6.0};
	std::vector<double> high = {1.5, 1.5, 0.99, 4.0, 4.0, 10.0};
	space->SetLow(low);
	space->SetHigh(high);
	return space;
}

Ptr<OpenGymDataContainer> FusionGymEnv::GetObservation()
{
	NS_LOG_FUNCTION(this);
	if (!m_agent || !m_qp) {
		std::vector<double> obs(7, 0.0);
		Ptr<OpenGymBoxContainer<double> > box = CreateObject<OpenGymBoxContainer<double> > ();
		box->SetData(obs);
		return box;
	}
	FusionAgent::NetworkState stateToUse;
	auto it = m_qpStates.find(m_qp);
	if (it != m_qpStates.end()) {
		stateToUse = it->second;
	} else {
		stateToUse = m_currentState;
		NS_LOG_WARN("State not found for QP in GetObservation, using current state");
	}
	std::vector<double> obs = StateToObservation(stateToUse);
	Ptr<OpenGymBoxContainer<double> > box = CreateObject<OpenGymBoxContainer<double> > ();
	box->SetData(obs);
	return box;
}

Ptr<OpenGymDataContainer> FusionGymEnv::GetReward()
{
	NS_LOG_FUNCTION(this);
	
	Ptr<OpenGymDoubleContainer> reward = CreateObject<OpenGymDoubleContainer> ();
	reward->SetValue(m_currentReward);
	
	return reward;
}

bool FusionGymEnv::GetGameOver()
{
	NS_LOG_FUNCTION(this);
	if (m_qp && m_qp->IsFinished()) {
		m_gameOver = true;
	}
	return m_gameOver;
}

std::string FusionGymEnv::GetExtraInfo()
{
	NS_LOG_FUNCTION(this);
	
	std::stringstream ss;
	ss << "episode_reward:" << m_episodeReward << ","
	   << "episode_step:" << m_episodeStep << ","
	   << "thr:" << m_currentState.thr << ","
	   << "rtt:" << m_currentState.rtt;
	
	return ss.str();
}

bool FusionGymEnv::ExecuteActions(Ptr<OpenGymDataContainer> action)
{
	NS_LOG_FUNCTION(this << action);
	if (!m_agent || !m_qp) {
		return false;
	}
	FusionAgent::FusionParams params = ActionToParams(action);
	m_qpParams[m_qp] = params;
	m_agent->ApplyParameters(m_qp, params);
	m_agent->MarkParametersUpdated(m_qp, params);
	NS_LOG_DEBUG("Fusion params updated via RL for QP: delta=" << params.delta 
	             << " gamma=" << params.gamma 
	             << " beta=" << params.beta
	             << " mu1=" << params.mu1
	             << " mu2=" << params.mu2
	             << " mu3=" << params.mu3);
	return true;
}
#endif // NS3_OPENGYM

std::vector<double> FusionGymEnv::StateToObservation(const FusionAgent::NetworkState &state)
{
	NS_LOG_FUNCTION(this);
	std::vector<double> obs(7);
	obs[0] = state.thr;
	if (state.rtt_min > 0) {
		obs[1] = state.rtt / state.rtt_min;
	} else {
		obs[1] = 1.0;
	}
	obs[2] = state.thr_max;
	obs[3] = state.rtt_min;
	obs[4] = std::max(0.0, std::min(1.0, state.loss));
	obs[5] = 1.0 + state.delta_Rt;
	obs[5] = std::max(0.0, std::min(2.0, obs[5]));
	obs[6] = std::max(0.0, std::min(1.0, state.Occ));
	return obs;
}

#ifdef NS3_OPENGYM
FusionAgent::FusionParams FusionGymEnv::ActionToParams(Ptr<OpenGymDataContainer> action)
{
	NS_LOG_FUNCTION(this << action);
	FusionAgent::FusionParams params;
	Ptr<OpenGymBoxContainer<double> > box = DynamicCast<OpenGymBoxContainer<double> >(action);
	if (box == 0) {
		NS_LOG_ERROR("Invalid action container type");
		return params;
	}
	std::vector<double> actionVec = box->GetData();
	if (actionVec.size() < 6) {
		NS_LOG_ERROR("Invalid action vector size: " << actionVec.size());
		return params;
	}
	params.delta = std::max(0.5, std::min(1.5, actionVec[0]));
	params.gamma = std::max(0.5, std::min(1.5, actionVec[1]));
	params.beta = std::max(0.01, std::min(0.99, actionVec[2]));
	params.mu1 = std::max(2.1, std::min(4.0, actionVec[3]));
	params.mu2 = std::max(params.mu1, std::min(4.0, actionVec[4]));
	params.mu3 = std::max(6.0, std::min(10.0, actionVec[5]));
	return params;
}
#endif // NS3_OPENGYM

} /* namespace ns3 */

