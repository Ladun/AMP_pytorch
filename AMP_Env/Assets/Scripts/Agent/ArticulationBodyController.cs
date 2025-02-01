using System.Collections.Generic;
using System.Data;
using Unity.MLAgents;
using UnityEngine;
using UnityEngine.Serialization;
using static UnityEngine.GraphicsBuffer;


namespace AMP
{
    [System.Serializable]
    public class ArticulationBodyPart
    {
        // https://github.com/xbpeng/DeepMimic/blob/70e7c6b22b775bb9342d4e15e6ef0bd91a55c6c0/DeepMimicCore/sim/CtCtrlUtil.cpp#L7
        private const float gMaxPDExpVal = 2.0f * Mathf.PI;

        [Header("Body Part Info")]
        public ArticulationBody ab;

        public Vector3 lowerLim;
        public Vector3 upperLim;

        [Header("Ground & Target Contact")]
        [Space(10)]
        public GroundContact groundContact;

        public TargetContact targetContact;

        [FormerlySerializedAs("thisABController")]
        [HideInInspector] public ArticulationBodyController thisABController;

        public void Reset(Vector3 euler)
        {

            if (ab.jointType == ArticulationJointType.SphericalJoint)
            {
                ab.SetDriveRotation(Quaternion.Euler(euler)); 
                ab.resetJointPosition(ab.ToTargetRotationInReducedSpace(Quaternion.Euler(euler), false));

            }
            else if (ab.jointType == ArticulationJointType.RevoluteJoint)
            {
                ab.SetDriveTarget(ArticulationDriveAxis.X, euler.z);
                ab.resetJointPosition(euler.z * Mathf.Deg2Rad);
            }
            ab.linearVelocity = Vector3.zero;
            ab.angularVelocity = Vector3.zero;
        }

        public void SetJointTarget(List<float> f)
        {
            if (ab.jointType == ArticulationJointType.SphericalJoint)
            {
                Vector3 euler;

                euler.x = (f[0] + 1) * 0.5f * (ab.xDrive.upperLimit - ab.xDrive.lowerLimit) + ab.xDrive.lowerLimit;
                euler.y = (f[1] + 1) * 0.5f * (ab.yDrive.upperLimit - ab.yDrive.lowerLimit) + ab.yDrive.lowerLimit;
                euler.z = (f[2] + 1) * 0.5f * (ab.zDrive.upperLimit - ab.zDrive.lowerLimit) + ab.zDrive.lowerLimit;

                ab.SetDriveRotation(Quaternion.Euler(euler));
            }

            else if (ab.jointType == ArticulationJointType.RevoluteJoint)
            {
                float q = f[0] * Mathf.Rad2Deg;
                q = Mathf.Clamp(q, ab.xDrive.lowerLimit, ab.xDrive.upperLimit);
                ab.SetDriveTarget(ArticulationDriveAxis.X, q);
            }
        }

        public void SetJointTargetFromRotVector(List<float> f)
        {
            const float maxLen = gMaxPDExpVal;

            if (ab.jointType == ArticulationJointType.SphericalJoint)
            {
                Vector3 exp_map = new Vector3(f[0], f[1], f[2]);
                float len = exp_map.magnitude;
                if(len > maxLen)
                {
                    exp_map *= maxLen / len;
                }

                // ExpMapToQuaternion
                float theta = exp_map.magnitude;
                float outTheta = 0;
                Vector3 outAxis = new Vector3(0, 0, 1);
                if(theta > 1e-6)
                {
                    outAxis = exp_map / theta;
                    outTheta = Utils.NormlaizeAngle(theta);
                }

                float c = Mathf.Cos(outTheta / 2);
                float s = Mathf.Sin(outTheta / 2);
                Quaternion quat = new Quaternion(
                        s * outAxis.x,
                        s * outAxis.y,
                        s * outAxis.z,
                        c);

                ab.SetDriveRotation(quat);
            }

            else if(ab.jointType == ArticulationJointType.RevoluteJoint)
            {
                ab.SetDriveTarget(ArticulationDriveAxis.X, f[0] * Mathf.Rad2Deg);
            }

        }

        public bool IsPenalizable()
        {
            if (ab.jointType == ArticulationJointType.SphericalJoint)
            {
                if (ab.twistLock == ArticulationDofLock.LimitedMotion ||
                    ab.swingYLock == ArticulationDofLock.LimitedMotion ||
                    ab.swingZLock == ArticulationDofLock.LimitedMotion)
                    return true;
            }
            else if (ab.jointType == ArticulationJointType.RevoluteJoint)
            {
                if (ab.twistLock == ArticulationDofLock.LimitedMotion)
                    return true;
            }
            return false;
        }

        public float GetBoundsPenalty()
        {
            float penalty = 0;
            if (ab.jointType == ArticulationJointType.SphericalJoint)
            {
                int cnt = 0;
                if(ab.twistLock == ArticulationDofLock.LimitedMotion)
                {
                    if (ab.xDrive.target < ab.xDrive.lowerLimit || ab.xDrive.target > ab.xDrive.upperLimit)
                        penalty += 1;
                    cnt++;
                }
                if (ab.swingYLock == ArticulationDofLock.LimitedMotion)
                {
                    if (ab.yDrive.target < ab.yDrive.lowerLimit || ab.yDrive.target > ab.yDrive.upperLimit)
                        penalty += 1;
                    cnt++;
                }
                if (ab.swingZLock == ArticulationDofLock.LimitedMotion)
                {
                    if (ab.zDrive.target < ab.zDrive.lowerLimit || ab.zDrive.target > ab.zDrive.upperLimit)
                        penalty += 1;
                    cnt++;
                }
                if (cnt > 0)
                    penalty /= cnt;
            }
            else if (ab.jointType == ArticulationJointType.RevoluteJoint)
            {
                if (ab.twistLock == ArticulationDofLock.LimitedMotion)
                {
                    if (ab.xDrive.target < ab.xDrive.lowerLimit || ab.xDrive.target > ab.xDrive.upperLimit)
                        penalty += 1;
                }
            }
            return penalty;
        }
    }

    public class ArticulationBodyController : MonoBehaviour
    {
        [Header("Drive info")]
        public float stiffness = 100f; 
        public float damping = 10f;    


        [HideInInspector] public SortedDictionary<int, ArticulationBodyPart> bodyPartsDict = new SortedDictionary<int, ArticulationBodyPart>();
        [HideInInspector] public List<ArticulationBodyPart> bodyPartsList = new List<ArticulationBodyPart>();

        public void SetupBodyPart(int key, Transform t)
        {
            ArticulationBody ab = t.GetComponent<ArticulationBody>();
            var bp = new ArticulationBodyPart
            {
                ab = ab
            };

            bp.groundContact = t.GetComponent<GroundContact>();
            if (!bp.groundContact)
            {
                bp.groundContact = t.gameObject.AddComponent<GroundContact>();
                bp.groundContact.agent = gameObject.GetComponent<Agent>();
            }
            else
            {
                bp.groundContact.agent = gameObject.GetComponent<Agent>();
            }

            bp.thisABController = this;
            bodyPartsDict.Add(key, bp);
            bodyPartsList.Add(bp);
        }

        public void ResetState()
        {
            bodyPartsDict.Clear();
            bodyPartsList.Clear();
        }
    }

}