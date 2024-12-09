using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEditor;
using UnityEditor.TerrainTools;
using Unity.MLAgents.Policies;
using PlasticGui.WorkspaceWindow.PendingChanges;

namespace AMP
{
    [CustomEditor(typeof(DeepMinicSkeleton))]
    public class DeepMinicSkeletonEditor : Editor
    {

        public override void OnInspectorGUI()
        {
            DeepMinicSkeleton c = (DeepMinicSkeleton)target;

            DrawDefaultInspector();

            
            if (c.HasSkeleton())
            {
                EditorGUILayout.Space(10);
                c.UpdateObs();
                Skeleton.Observastion obs = c.Obs;
                EditorGUILayout.LabelField($"State Size: {obs.GetObsSize()} ");
                EditorGUILayout.LabelField($"Num of joints: {c.NumOfJoints} ");
                EditorGUILayout.LabelField($"Num of dofs: {c.numOfDofs} ");
            }

            if (GUILayout.Button("Generate Skeleton", GUILayout.Width(200)))
            {
                c.CreateSkeleton();
                c.ConfigureJoints();
            }

        }
    }
}