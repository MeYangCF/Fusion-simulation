#include "fusion-agent.h"
#include "fusion-gym-env.h"
#include <ns3/simulator.h>
#include <ns3/log.h>
#include <algorithm>
#include <cmath>

NS_LOG_COMPONENT_DEFINE("FusionAgent");

namespace ns3 {

NS_OBJECT_ENSURE_REGISTERED(FusionAgent);

TypeId FusionAgent::GetTypeId (void)
{
	static TypeId tid = TypeId ("ns3::FusionAgent")
		.SetParent<Object> ()
		.AddAttribute("MonitoringInterval",
				"Monitoring interval for collecting statistics",
				TimeValue(MicroSeconds(100)),
				MakeTimeAccessor(&FusionAgent::m_monitoringInterval),
				MakeTimeChecker())
		.AddAttribute("Omega1",
				"Weight for packet loss penalty",
				DoubleValue(0.1),
				MakeDoubleAccessor(&FusionAgent::m_omega1),
				MakeDoubleChecker<double>())
		.AddAttribute("Omega2",
				"Weight for rate change",
				DoubleValue(0.1),
				MakeDoubleAccessor(&FusionAgent::m_omega2),
				MakeDoubleChecker<double>())
		.AddAttribute("Omega3",
				"Weight for buffer occupancy",
				DoubleValue(0.1),
				MakeDoubleAccessor(&FusionAgent::m_omega3),
				MakeDoubleChecker<double>())
		.AddAttribute("Omega4",
				"RTT threshold multiplier",
				DoubleValue(1.5),
				MakeDoubleAccessor(&FusionAgent::m_omega4),
				MakeDoubleChecker<double>())
		.AddAttribute("ParamUpdateTimeout",
				"Timeout for parameter update from Python (if exceeded, use fallback)",
				TimeValue(MilliSeconds(10)),
				MakeTimeAccessor(&FusionAgent::m_paramUpdateTimeout),
				MakeTimeChecker())
		;
	return tid;
}

FusionAgent::FusionAgent()
	: m_monitoringInterval(MicroSeconds(100)),
	  m_omega1(0.1), m_omega2(0.1), m_omega3(0.1), m_omega4(1.5),
	  m_useGym(false), m_gymEnv(0),
	  m_paramUpdateTimeout(MilliSeconds(10))
{
	NS_LOG_FUNCTION(this);
}

FusionAgent::~FusionAgent()
{
	NS_LOG_FUNCTION(this);
}

void FusionAgent::Initialize()
{
	NS_LOG_FUNCTION(this);
	m_qpStates.clear();
	m_qpParamStates.clear();
	for (auto &pair : m_qpMonitoringEvents) {
		if (pair.second.IsRunning()) {
			Simulator::Cancel(pair.second);
		}
	}
	m_qpMonitoringEvents.clear();
}

void FusionAgent::CollectStatistics(Ptr<RdmaQueuePair> qp, NetworkState &state)
{
	NS_LOG_FUNCTION(this << qp);
	double currentRate = qp->m_rate.GetBitRate() * 1e-9;
	state.thr = currentRate;
	if (state.thr > state.thr_max || state.thr_max == 0) {
		state.thr_max = state.thr;
	}
	double rtt = 0.0;
	if (qp->fusion.m_lastRtt > 0) {
		rtt = qp->fusion.m_lastRtt * 1e-9;
	} else {
		rtt = qp->m_baseRtt * 1e-9;
	}
	state.rtt = rtt;
	double minRtt = 0.0;
	if (qp->fusion.m_minRtt > 0) {
		minRtt = qp->fusion.m_minRtt * 1e-9;
	} else {
		minRtt = rtt;
	}
	if (minRtt > 0 && (state.rtt_min == 0 || minRtt < state.rtt_min)) {
		state.rtt_min = minRtt;
	}
	double targetRate = qp->fusion.m_targetRate.GetBitRate() * 1e-9;
	if (m_qpStates.find(qp) != m_qpStates.end()) {
		const NetworkState &prevState = m_qpStates[qp];
		double prevTargetRate = prevState.prev_target_rate;
		if (prevTargetRate > 0) {
			state.delta_Rt = (targetRate - prevTargetRate) / std::min(prevTargetRate, targetRate);
		}
	}
	state.prev_target_rate = targetRate;
	state.loss = 0.0;
	if (qp->fusion.m_packetsSent > 0) {
		if (qp->fusion.m_packetsSent > qp->fusion.m_packetsAcked) {
			uint32_t lostPackets = qp->fusion.m_packetsSent - qp->fusion.m_packetsAcked;
			state.loss = (double)lostPackets / (double)qp->fusion.m_packetsSent;
		}
		double nackLoss = 0.0;
		if (qp->fusion.m_packetsSent > 0) {
			nackLoss = (double)qp->fusion.m_nackCount / (double)qp->fusion.m_packetsSent;
		}
		state.loss = std::max(state.loss, nackLoss);
		state.loss = std::max(0.0, std::min(1.0, state.loss));
	}
	state.Occ = 0.0;
	if (qp->fusion.m_totalPackets > 0) {
		state.Occ = (double)qp->fusion.m_ecnMarkedPackets / (double)qp->fusion.m_totalPackets;
		state.Occ = std::max(0.0, std::min(1.0, state.Occ));
	}
	Time now = Simulator::Now();
	if (qp->fusion.m_lastStatsReset == Time(0)) {
		qp->fusion.m_lastStatsReset = now;
	} else if (now - qp->fusion.m_lastStatsReset >= m_monitoringInterval) {
		qp->fusion.m_ecnMarkedPackets = 0;
		qp->fusion.m_totalPackets = 0;
		qp->fusion.m_nackCount = 0;
		qp->fusion.m_packetsSent = 0;
		qp->fusion.m_packetsAcked = 0;
		qp->fusion.m_lastStatsReset = now;
	}
	m_qpStates[qp] = state;
}

FusionAgent::FusionParams FusionAgent::GenerateParameters(const NetworkState &state)
{
	NS_LOG_FUNCTION(this);
	NetworkState normalizedState = state;
	NormalizeState(normalizedState);
	FusionParams params = AdjustParameters(normalizedState);
	if (m_useGym && m_gymEnv) {
		NS_LOG_DEBUG("GenerateParameters called in Gym mode - this is fallback only. "
		             << "RL parameters should be set via ExecuteActions.");
	}
	return params;
}

double FusionAgent::CalculateReward(const NetworkState &state)
{
	NS_LOG_FUNCTION(this);
	double rtt_prime;
	if (state.rtt_min > 0 && state.rtt_min <= state.rtt && state.rtt <= m_omega4 * state.rtt_min) {
		rtt_prime = state.rtt_min;
	} else {
		rtt_prime = state.rtt;
	}
	if (rtt_prime <= 0) {
		rtt_prime = 1e-9;
	}
	double power_term = 0.0;
	if (state.thr_max > 0 && state.rtt_min > 0) {
		double normalized_power = (state.thr - m_omega1 * state.loss) / rtt_prime;
		double max_power = state.thr_max / state.rtt_min;
		if (max_power > 0) {
			power_term = normalized_power / max_power;
		}
	}
	double reward = power_term + m_omega2 * state.delta_Rt - m_omega3 * state.Occ;
	return reward;
}

void FusionAgent::UpdateAgent(const NetworkState &state, double reward)
{
	NS_LOG_FUNCTION(this << reward);
}

void FusionAgent::SetMonitoringInterval(Time interval)
{
	NS_LOG_FUNCTION(this << interval);
	m_monitoringInterval = interval;
}

Time FusionAgent::GetMonitoringInterval() const
{
	NS_LOG_FUNCTION(this);
	return m_monitoringInterval;
}

void FusionAgent::NormalizeState(NetworkState &state)
{
	NS_LOG_FUNCTION(this);
	if (state.thr_max > 0) {
		state.thr = state.thr / state.thr_max;
	}
	if (state.rtt_min > 0 && state.rtt > 0) {
		state.rtt = state.rtt / state.rtt_min;
	}
}

FusionAgent::FusionParams FusionAgent::AdjustParameters(const NetworkState &state)
{
	NS_LOG_FUNCTION(this);
	FusionParams params = m_defaultParams;
	if (state.Occ > 0.8) {
		params.delta = 1.5;
	} else if (state.Occ < 0.2) {
		params.delta = 0.5;
	} else {
		params.delta = 1.0;
	}
	if (state.thr < 0.5) {
		params.gamma = 0.5;
	} else {
		params.gamma = 1.0;
	}
	if (state.delta_Rt < -0.1) {
		params.beta = 0.05;
	} else {
		params.beta = 0.1;
	}
	if (state.Occ > 0.7 || state.loss > 0.1) {
		params.mu1 = 2.5;
		params.mu2 = 3.5;
	} else {
		params.mu1 = 2.1;
		params.mu2 = 3.0;
	}
	if (state.rtt > 1.5 * state.rtt_min) {
		params.mu3 = 10.0;
	} else {
		params.mu3 = 8.0;
	}
	params.delta = std::max(0.5, std::min(1.5, params.delta));
	params.gamma = std::max(0.5, std::min(1.5, params.gamma));
	params.beta = std::max(0.01, std::min(0.99, params.beta));
	params.mu1 = std::max(2.1, std::min(4.0, params.mu1));
	params.mu2 = std::max(params.mu1, std::min(4.0, params.mu2));
	params.mu3 = std::max(6.0, std::min(10.0, params.mu3));
	return params;
}

void FusionAgent::SetUseGym(bool useGym)
{
	NS_LOG_FUNCTION(this << useGym);
	m_useGym = useGym;
}

bool FusionAgent::GetUseGym() const
{
	NS_LOG_FUNCTION(this);
	return m_useGym;
}

void FusionAgent::SetGymEnv(Ptr<FusionGymEnv> env)
{
	NS_LOG_FUNCTION(this << env);
	m_gymEnv = env;
	if (env) {
		env->SetFusionAgent(this);
		m_useGym = true;
	}
}

Ptr<FusionGymEnv> FusionAgent::GetGymEnv() const
{
	NS_LOG_FUNCTION(this);
	return m_gymEnv;
}

void FusionAgent::StartMonitoring(Ptr<RdmaQueuePair> qp)
{
	NS_LOG_FUNCTION(this << qp);
	if (!qp) {
		return;
	}
	if (m_qpParamStates.find(qp) == m_qpParamStates.end()) {
		m_qpParamStates[qp] = QpParamState();
		m_qpParamStates[qp].lastParams = m_defaultParams;
	}
	if (m_gymEnv) {
		m_gymEnv->SetQueuePair(qp);
	}
	ScheduleNextMonitoring(qp);
}

void FusionAgent::ScheduleNextMonitoring(Ptr<RdmaQueuePair> qp)
{
	NS_LOG_FUNCTION(this << qp);
	if (qp && !qp->IsFinished()) {
		auto it = m_qpMonitoringEvents.find(qp);
		if (it != m_qpMonitoringEvents.end() && it->second.IsRunning()) {
			Simulator::Cancel(it->second);
		}
		EventId event = Simulator::Schedule(m_monitoringInterval,
		                                    &FusionAgent::MonitoringIntervalCallback,
		                                    this, qp);
		m_qpMonitoringEvents[qp] = event;
	}
}

void FusionAgent::MonitoringIntervalCallback(Ptr<RdmaQueuePair> qp)
{
	NS_LOG_FUNCTION(this << qp);
	if (!qp || qp->IsFinished()) {
		return;
	}
	CheckPendingParameters(qp);
	NetworkState state;
	CollectStatistics(qp, state);
	if (m_useGym && m_gymEnv) {
		m_gymEnv->SetQueuePair(qp);
		if (m_qpParamStates.find(qp) == m_qpParamStates.end()) {
			m_qpParamStates[qp] = QpParamState();
			m_qpParamStates[qp].lastParams = m_defaultParams;
		}
		m_qpParamStates[qp].paramsPending = true;
		m_qpParamStates[qp].pendingSince = Simulator::Now();
		m_gymEnv->NotifyMonitoringInterval(qp, state);
		Simulator::Schedule(m_paramUpdateTimeout, &FusionAgent::CheckPendingParameters, this, qp);
	} else {
		FusionParams params = GenerateParameters(state);
		ApplyParameters(qp, params);
		MarkParametersUpdated(qp, params);
	}
	ScheduleNextMonitoring(qp);
}

void FusionAgent::CheckPendingParameters(Ptr<RdmaQueuePair> qp)
{
	NS_LOG_FUNCTION(this << qp);
	if (!qp || qp->IsFinished()) {
		return;
	}
	auto it = m_qpParamStates.find(qp);
	if (it == m_qpParamStates.end()) {
		return;
	}
	QpParamState &paramState = it->second;
	if (paramState.paramsPending) {
		Time elapsed = Simulator::Now() - paramState.pendingSince;
		if (elapsed >= m_paramUpdateTimeout) {
			NS_LOG_WARN("Parameter update timeout for QP. Using fallback parameters.");
			FusionParams fallbackParams = paramState.lastParams;
			if (paramState.lastUpdateTime == Time(0)) {
				fallbackParams = m_defaultParams;
			}
			if (paramState.paramsPending) {
				ApplyParameters(qp, fallbackParams);
				paramState.paramsPending = false;
			}
		}
	}
}

void FusionAgent::ApplyParameters(Ptr<RdmaQueuePair> qp, const FusionParams &params)
{
	NS_LOG_FUNCTION(this << qp);
	if (!qp) {
		return;
	}
	qp->fusion.delta = std::max(0.5, std::min(1.5, params.delta));
	qp->fusion.gamma = std::max(0.5, std::min(1.5, params.gamma));
	qp->fusion.beta = std::max(0.01, std::min(0.99, params.beta));
	qp->fusion.mu1 = std::max(2.1, std::min(4.0, params.mu1));
	qp->fusion.mu2 = std::max(qp->fusion.mu1, std::min(4.0, params.mu2));
	qp->fusion.mu3 = std::max(6.0, std::min(10.0, params.mu3));
	NS_LOG_DEBUG("Applied Fusion parameters: delta=" << params.delta 
	             << " gamma=" << params.gamma 
	             << " beta=" << params.beta
	             << " mu1=" << params.mu1
	             << " mu2=" << params.mu2
	             << " mu3=" << params.mu3);
}

void FusionAgent::MarkParametersUpdated(Ptr<RdmaQueuePair> qp, const FusionParams &params)
{
	NS_LOG_FUNCTION(this << qp);
	
	if (!qp) {
		return;
	}
	
	if (m_qpParamStates.find(qp) == m_qpParamStates.end()) {
		m_qpParamStates[qp] = QpParamState();
	}
	
	QpParamState &paramState = m_qpParamStates[qp];
	paramState.lastParams = params;
	paramState.lastUpdateTime = Simulator::Now();
	paramState.paramsPending = false;
}

} /* namespace ns3 */

