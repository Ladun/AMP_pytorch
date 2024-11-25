using AMP;
using Newtonsoft.Json;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using UnityEngine;

public class MotionDatabase : MonoBehaviour
{
    #region Class for parse json
    public class Motions
    {
        public float Weight;
        public string File;
    }
    #endregion

    public enum DataType { DeepMimic };

    public DataType type;

    public class DatasetData
    {
        public Motions[] Motions;
    }

    public MotionParser parser;
    public string datasetFile;

    private Dictionary<string, List<MotionFrameData>> motions = new Dictionary<string, List<MotionFrameData>>();

    public bool HasMotion
    {
        get {  return motions != null && motions.Count > 0; }
    }

    public void LoadDataset(bool forced=false)
    {
        if (motions.Count > 0 && !forced)
            return;

        InitParser();
        motions.Clear();

        string path = Path.Combine(Utils.GetCurrentPath(), datasetFile);
        string text = Utils.ReadTextFile(path);
        if (string.IsNullOrEmpty(text))
        {
            Debug.LogWarning($"Wrong text {datasetFile}");
            return;
        }

        DatasetData dataset = JsonConvert.DeserializeObject<DatasetData>(text);
        foreach (Motions data in dataset.Motions)
        {
            var m = parser.LoadData(Path.Combine(Utils.GetCurrentPath(), data.File));
            motions.Add(data.File, m);
        }
    }

    public List<MotionFrameData> GetMotionData(string motionKey)
    {
        if (motions.ContainsKey(motionKey))
            return motions[motionKey];
        return null;
    }

    public string[] GetMotionKeys()
    {
        return motions.Keys.ToArray();
    }

    public MotionFrameData GetRandomMotionData()
    {
        var motionKeys = motions.Keys.ToList();
        var randomMotion = motions[motionKeys[Random.Range(0, motionKeys.Count)]];


        return randomMotion[Random.Range(0, randomMotion.Count)];
    }

    private void InitParser()
    {
        parser = type switch
        {
            DataType.DeepMimic => new DeepMimicMotionParser(),
            _ => null
        };
    }
}
