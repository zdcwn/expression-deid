// Reassemble the unchanged GPU executable from GitHub-sized assets, then launch it.
using System;
using System.ComponentModel;
using System.Diagnostics;
using System.IO;
using System.Security.Cryptography;
using System.Text;
using System.Threading;
using System.Windows.Forms;

internal static class ReassembleLauncher {
    private static bool Ready(string destination) {
        string marker=destination+".ready";
        return File.Exists(destination) && new FileInfo(destination).Length==PackageInfo.Size
            && File.Exists(marker) && File.ReadAllText(marker)==PackageInfo.Hash;
    }

    private static void Assemble(string destination,BackgroundWorker worker) {
        using(var gate=new Mutex(false,"Local\\ExpressionLabParts-"+PackageInfo.Hash.Substring(0,16))) {
            try { gate.WaitOne(); } catch(AbandonedMutexException) { }
            try {
                if(Ready(destination)) return;
                string folder=AppDomain.CurrentDomain.BaseDirectory;
                for(int i=0;i<PackageInfo.Parts.Length;i++) {
                    string part=Path.Combine(folder,PackageInfo.Parts[i]);
                    if(!File.Exists(part) || new FileInfo(part).Length!=PackageInfo.PartSizes[i])
                        throw new IOException("缺少或未下载完整的分卷："+PackageInfo.Parts[i]
                            +"\r\n请将全部 GPU 分卷和本启动文件放在同一个文件夹。\r\n无需手动解压或合并。");
                }
                Directory.CreateDirectory(Path.GetDirectoryName(destination));
                string temporary=destination+".partial";
                byte[] buffer=new byte[1024*1024]; long copied=0;
                using(var hash=SHA256.Create()) {
                    using(var output=new FileStream(temporary,FileMode.Create,FileAccess.Write,FileShare.None)) {
                        foreach(string name in PackageInfo.Parts) {
                            using(var input=File.OpenRead(Path.Combine(folder,name))) {
                                int count;
                                while((count=input.Read(buffer,0,buffer.Length))>0) {
                                    output.Write(buffer,0,count); hash.TransformBlock(buffer,0,count,buffer,0);
                                    copied+=count; worker.ReportProgress((int)(copied*100/PackageInfo.Size));
                                }
                            }
                        }
                        hash.TransformFinalBlock(new byte[0],0,0);
                    }
                    string actual=BitConverter.ToString(hash.Hash).Replace("-","").ToLowerInvariant();
                    if(copied!=PackageInfo.Size || actual!=PackageInfo.Hash)
                        throw new InvalidDataException("GPU 分卷校验失败，请重新下载分卷后再启动。");
                }
                if(File.Exists(destination)) File.Replace(temporary,destination,null);
                else File.Move(temporary,destination);
                File.WriteAllText(destination+".ready",PackageInfo.Hash,Encoding.ASCII);
            } finally { gate.ReleaseMutex(); }
        }
    }

    private static string Quote(string value) {
        var result=new StringBuilder("\""); int slashes=0;
        foreach(char c in value) {
            if(c=='\\') { slashes++; continue; }
            if(c=='\"') result.Append('\\',slashes*2+1);
            else result.Append('\\',slashes);
            result.Append(c); slashes=0;
        }
        result.Append('\\',slashes*2); result.Append('"'); return result.ToString();
    }

    [STAThread]
    public static int Main(string[] args) {
        Application.EnableVisualStyles();
        try {
            string local=Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData);
            string destination=Path.Combine(local,"ExpressionLab","packages",PackageInfo.Hash.Substring(0,16),"ExpressionLab-GPU.exe");
            if(!Ready(destination)) {
                Exception failure=null;
                using(var form=new Form()) {
                    form.Text="Expression Lab · GPU"; form.ClientSize=new System.Drawing.Size(510,150);
                    form.StartPosition=FormStartPosition.CenterScreen; form.FormBorderStyle=FormBorderStyle.FixedDialog;
                    form.MaximizeBox=false; form.MinimizeBox=false; form.ControlBox=false;
                    var text=new Label(); text.Text="正在准备 GPU 版软件…\r\n首次启动自动合并并校验分卷，无需安装。";
                    text.SetBounds(25,25,460,60); text.Font=new System.Drawing.Font("Microsoft YaHei UI",10);
                    var progress=new ProgressBar(); progress.SetBounds(25,105,460,20);
                    form.Controls.Add(text); form.Controls.Add(progress);
                    var worker=new BackgroundWorker(); worker.WorkerReportsProgress=true;
                    worker.DoWork+=delegate { Assemble(destination,worker); };
                    worker.ProgressChanged+=delegate(object sender,ProgressChangedEventArgs e) { progress.Value=e.ProgressPercentage; };
                    worker.RunWorkerCompleted+=delegate(object sender,RunWorkerCompletedEventArgs e) { failure=e.Error; form.Close(); };
                    form.Shown+=delegate { worker.RunWorkerAsync(); }; Application.Run(form);
                }
                if(failure!=null) throw failure;
            }
            var arguments=new StringBuilder();
            foreach(string arg in args) { if(arguments.Length>0) arguments.Append(' '); arguments.Append(Quote(arg)); }
            var info=new ProcessStartInfo(destination,arguments.ToString());
            info.UseShellExecute=false; info.WorkingDirectory=Path.GetDirectoryName(destination);
            var process=Process.Start(info);
            if(Array.IndexOf(args,"--check-package")>=0) { process.WaitForExit(); return process.ExitCode; }
            return 0;
        } catch(Exception error) {
            MessageBox.Show(error.Message,"Expression Lab · GPU",MessageBoxButtons.OK,MessageBoxIcon.Error); return 1;
        }
    }
}
