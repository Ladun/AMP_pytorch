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
            if (ab.jointType == ArticulationJointType.SphericalJoint)
            {
                float x = f[0];
                float y = f[1];
                float z = f[2];

                float theta = Mathf.Sqrt(x * x + y * y + z * z);
                Quaternion rot = Quaternion.identity;
                if (theta != 0)
                {
                    x = (x / theta);
                    y = (y / theta);
                    z = (z / theta);
                    rot = new Quaternion(x * Mathf.Sin(theta / 2),
                                         y * Mathf.Sin(theta / 2),
                                         z * Mathf.Sin(theta / 2),
                                         Mathf.Cos(theta / 2));
                }
                Vector3 euler = rot.eulerAngles;

                euler.x = Mathf.Clamp(euler.x, ab.xDrive.lowerLimit, ab.xDrive.upperLimit);
                euler.y = Mathf.Clamp(euler.y, ab.yDrive.lowerLimit, ab.yDrive.upperLimit);
                euler.z = Mathf.Clamp(euler.z, ab.zDrive.lowerLimit, ab.zDrive.upperLimit);

                ab.SetDriveRotation(Quaternion.Euler(euler));
            }

            else if(ab.jointType == ArticulationJointType.RevoluteJoint)
            {
                float q = f[0]* Mathf.Rad2Deg;
                q = Mathf.Clamp(q, ab.xDrive.lowerLimit, ab.xDrive.upperLimit);
                ab.SetDriveTarget(ArticulationDriveAxis.X, q);
            }

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