// First-run extraction uses only the .NET Framework included with Windows 10/11.
using System;
using System.ComponentModel;
using System.Diagnostics;
using System.IO;
using System.IO.Compression;
using System.Reflection;
using System.Text;
using System.Threading;
using System.Windows.Forms;

internal sealed class PackageStream : Stream {
    private readonly Stream source;
    private readonly long start, length;
    private long position;
    public PackageStream(Stream source, long start, long length) { this.source=source; this.start=start; this.length=length; }
    public override bool CanRead { get { return true; } }
    public override bool CanSeek { get { return true; } }
    public override bool CanWrite { get { return false; } }
    public override long Length { get { return length; } }
    public override long Position { get { return position; } set { Seek(value,SeekOrigin.Begin); } }
    public override int Read(byte[] buffer,int offset,int count) {
        source.Position=start+position;
        int n=source.Read(buffer,offset,(int)Math.Min(count,length-position));
        position+=n; return n;
    }
    public override long Seek(long offset,SeekOrigin origin) {
        long next=origin==SeekOrigin.Begin ? offset : origin==SeekOrigin.Current ? position+offset : length+offset;
        if(next<0 || next>length) throw new IOException("Invalid package offset");
        return position=next;
    }
    public override void Flush() { }
    public override void SetLength(long value) { throw new NotSupportedException(); }
    public override void Write(byte[] buffer,int offset,int count) { throw new NotSupportedException(); }
}

internal static class PortableLauncher {
    private const string Magic="EXPRESSIONLABPKG1";
    private static readonly string Source=Assembly.GetExecutingAssembly().Location;
    private static long archiveStart,archiveSize;
    private static string packageHash;

    private static void ReadFooter() {
        using(var input=File.OpenRead(Source)) {
            long footer=input.Length-(16+64+Magic.Length);
            if(footer<0) throw new InvalidDataException("Incomplete application package");
            input.Position=footer;
            using(var reader=new BinaryReader(input,Encoding.ASCII)) {
                archiveStart=reader.ReadInt64(); archiveSize=reader.ReadInt64();
                packageHash=Encoding.ASCII.GetString(reader.ReadBytes(64));
                if(Encoding.ASCII.GetString(reader.ReadBytes(Magic.Length))!=Magic || archiveStart<1 || archiveSize<1 || archiveStart+archiveSize!=footer)
                    throw new InvalidDataException("Application package is incomplete. Copy the complete file again.");
                foreach(char c in packageHash)
                    if(!Uri.IsHexDigit(c)) throw new InvalidDataException("Invalid application package hash");
            }
        }
    }

    private static bool Ready(string cache) {
        string app=Path.Combine(cache,"ExpressionLab");
        return File.Exists(Path.Combine(cache,"ready.txt")) && File.Exists(Path.Combine(app,"ExpressionLab.exe"))
            && File.Exists(Path.Combine(app,"_internal","package_manifest.json"))
            && File.Exists(Path.Combine(app,"_internal","python313.dll"));
    }

    private static void Extract(string cache,BackgroundWorker worker) {
        using(var gate=new Mutex(false,"Local\\ExpressionLab-"+packageHash.Substring(0,16))) {
            try { gate.WaitOne(); } catch(AbandonedMutexException) { }
            try {
                if(Ready(cache)) return;
                Directory.CreateDirectory(cache);
                string prefix=Path.GetFullPath(cache)+Path.DirectorySeparatorChar;
                using(var source=File.OpenRead(Source))
                using(var region=new PackageStream(source,archiveStart,archiveSize))
                using(var zip=new ZipArchive(region,ZipArchiveMode.Read)) {
                    int index=0;
                    foreach(var entry in zip.Entries) {
                        string target=Path.GetFullPath(Path.Combine(cache,entry.FullName));
                        if(!target.StartsWith(prefix,StringComparison.OrdinalIgnoreCase))
                            throw new InvalidDataException("Unsafe application resource path");
                        if(String.IsNullOrEmpty(entry.Name)) Directory.CreateDirectory(target);
                        else {
                            Directory.CreateDirectory(Path.GetDirectoryName(target));
                            using(var input=entry.Open())
                            using(var output=new FileStream(target,FileMode.Create,FileAccess.Write,FileShare.None)) input.CopyTo(output);
                            if(new FileInfo(target).Length!=entry.Length) throw new InvalidDataException("Incomplete resource extraction");
                        }
                        worker.ReportProgress(++index*100/Math.Max(1,zip.Entries.Count));
                    }
                }
                File.WriteAllText(Path.Combine(cache,"ready.txt"),packageHash,Encoding.ASCII);
            } finally { gate.ReleaseMutex(); }
        }
    }

    private static string Quote(string value) {
        var result=new StringBuilder("\""); int slashes=0;
        foreach(char c in value) {
            if(c=='\\') { slashes++; continue; }
            if(c=='\"') { result.Append('\\',slashes*2+1); result.Append(c); }
            else { result.Append('\\',slashes); result.Append(c); }
            slashes=0;
        }
        result.Append('\\',slashes*2); result.Append('"'); return result.ToString();
    }

    [STAThread]
    public static int Main(string[] args) {
        AppContext.SetSwitch("Switch.System.IO.UseLegacyPathHandling",false);
        AppContext.SetSwitch("Switch.System.IO.BlockLongPaths",false);
        Application.EnableVisualStyles();
        try {
            ReadFooter();
            string local=Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData);
            string cache=Path.Combine(local,"ExpressionLab","apps",packageHash.Substring(0,16));
            if(!Ready(cache)) {
                Exception failure=null;
                using(var form=new Form()) {
                    form.Text="Expression Lab"; form.ClientSize=new System.Drawing.Size(500,150);
                    form.StartPosition=FormStartPosition.CenterScreen; form.FormBorderStyle=FormBorderStyle.FixedDialog;
                    form.MaximizeBox=false; form.MinimizeBox=false; form.ControlBox=false;
                    var text=new Label(); text.Text="Preparing Expression Lab for the first launch...\r\nAll resources are included. No download is needed.";
                    text.SetBounds(25,25,450,60); text.Font=new System.Drawing.Font("Segoe UI",10);
                    var progress=new ProgressBar(); progress.SetBounds(25,105,450,20);
                    form.Controls.Add(text); form.Controls.Add(progress);
                    var worker=new BackgroundWorker(); worker.WorkerReportsProgress=true;
                    worker.DoWork+=delegate { Extract(cache,worker); };
                    worker.ProgressChanged+=delegate(object sender,ProgressChangedEventArgs e) { progress.Value=e.ProgressPercentage; };
                    worker.RunWorkerCompleted+=delegate(object sender,RunWorkerCompletedEventArgs e) { failure=e.Error; form.Close(); };
                    form.Shown+=delegate { worker.RunWorkerAsync(); };
                    Application.Run(form);
                }
                if(failure!=null) throw failure;
            }
            string executable=Path.Combine(cache,"ExpressionLab","ExpressionLab.exe");
            var arguments=new StringBuilder();
            foreach(string arg in args) { if(arguments.Length>0) arguments.Append(' '); arguments.Append(Quote(arg)); }
            var info=new ProcessStartInfo(executable,arguments.ToString());
            info.UseShellExecute=false; info.WorkingDirectory=Path.GetDirectoryName(executable);
            var process=Process.Start(info);
            if(Array.IndexOf(args,"--check-package")>=0) { process.WaitForExit(); return process.ExitCode; }
            return 0;
        } catch(Exception error) {
            MessageBox.Show("Expression Lab could not start.\r\n\r\n"+error.Message,"Expression Lab",MessageBoxButtons.OK,MessageBoxIcon.Error);
            return 1;
        }
    }
}
