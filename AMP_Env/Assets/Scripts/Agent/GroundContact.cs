using UnityEngine;
using Unity.MLAgents;

namespace AMP
{
    /// <summary>
    /// This class contains logic for locomotion agents with joints which might make contact with the ground.
    /// By attaching this as a component to those joints, their contact with the ground can be used as either
    /// an observation for that agent, and/or a means of punishing the agent for making undesirable contact.
    /// </summary>
    [DisallowMultipleComponent]
    public class GroundContact : MonoBehaviour
    {
        [HideInInspector] public Agent agent;

        [Header("Ground Check")] 
        public bool agentDoneOnGroundContact; // Whether to reset agent on ground contact.
        public float timeToAgentDoneOnGroundContact = 0f;

        public bool penalizeGroundContact; // Whether to penalize on contact.
        public float groundContactPenalty; // Penalty amount (ex: -1).
        public bool touchingGround;
        const string groundTag = "ground"; // Tag of ground object.


        /// <summary>
        /// Check for collision with ground, and optionally penalize agent.
        /// </summary>
        void OnCollisionEnter(Collision col)
        {
            if (col.transform.CompareTag(groundTag))
            {
                touchingGround = true;
                if (penalizeGroundContact)
                {
                    agent.SetReward(groundContactPenalty);
                }
            }
        }

        private void OnCollisionStay(Collision col)
        {
            if (col.transform.CompareTag(groundTag))
            {
                if (agentDoneOnGroundContact)
                {
                    if (timeToAgentDoneOnGroundContact > 0)
                    {
                        timeToAgentDoneOnGroundContact -= Time.deltaTime;
                    }
                    else
                    {
                        agent.EndEpisode();
                    }
                }
            }
        }

        /// <summary>
        /// Check for end of ground collision and reset flag appropriately.
        /// </summary>
        void OnCollisionExit(Collision other)
        {
            if (other.transform.CompareTag(groundTag))
            {
                touchingGround = false;
            }
        }
    }
}
