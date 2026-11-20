// Index_RDA.ets - HarmonyOS radar range/velocity/azimuth dashboard.
// Copy this entire file to entry/src/main/ets/pages/Index.ets
// DevEco Emulator: 10.0.2.2 normally points to the Windows host.
// If this is not reachable in your simulator, use your Windows LAN IPv4.
import { http } from '@kit.NetworkKit';

interface RadarTarget {
  range_m: number;
  velocity_mps: number;
  azimuth_deg: number;
  x_m: number;
  y_m: number;
  peak_db: number;
}

interface RadarFrame {
  frame_id: number;
  weather: string;
  angle_method?: string;
  targets: RadarTarget[];
}

@Entry
@Component
struct Index {
  @State frameId: number = 0;
  @State weather: string = '--';
  @State targets: RadarTarget[] = [];
  @State status: string = '等待连接';

  // MATLAB -> Windows Python: 127.0.0.1:8765; emulator -> Windows: 10.0.2.2
  private serverUrl: string = 'http://10.0.2.2:8765/api/frame';
  private timerId: number = -1;
  private requesting: boolean = false;
  private canvasReady: boolean = false;
  private settings: RenderingContextSettings = new RenderingContextSettings(true);
  private context: CanvasRenderingContext2D = new CanvasRenderingContext2D(this.settings);

  aboutToAppear(): void {
    this.fetchLatest();
    this.timerId = setInterval(() => {
      this.fetchLatest();
    }, 2000);
  }

  aboutToDisappear(): void {
    if (this.timerId !== -1) {
      clearInterval(this.timerId);
      this.timerId = -1;
    }
  }

  private weatherZH(value: string): string {
    if (value === 'clear') { return '晴天'; }
    if (value === 'rain') { return '雨天'; }
    if (value === 'fog') { return '雾天'; }
    return value;
  }

  private fetchLatest(): void {
    if (this.requesting) { return; }
    this.requesting = true;
    const req = http.createHttp();

    req.request(this.serverUrl, {
      method: http.RequestMethod.GET,
      expectDataType: http.HttpDataType.STRING,
      usingCache: false,
      connectTimeout: 5000,
      readTimeout: 5000
    }).then((res: http.HttpResponse) => {
      if (res.responseCode !== 200) {
        this.status = 'HTTP状态：' + res.responseCode;
        return;
      }
      const frame: RadarFrame = JSON.parse(res.result as string) as RadarFrame;
      if (!Array.isArray(frame.targets)) {
        this.status = '数据格式错误：targets不是数组';
        return;
      }
      // Avoid showing false azimuth=0 when old 2D-only frames remain on server.
      if (frame.targets.length > 0 &&
        (typeof frame.targets[0].azimuth_deg !== 'number' ||
          typeof frame.targets[0].x_m !== 'number' ||
          typeof frame.targets[0].y_m !== 'number')) {
        this.status = '检测到旧版数据：请运行新版RDA MATLAB';
        this.targets = [];
        this.drawRadarMap();
        return;
      }
      this.frameId = frame.frame_id;
      this.weather = frame.weather;
      this.targets = frame.targets;
      this.status = '已连接 · CFAR + 8天线DBF';
      this.drawRadarMap();
    }).catch((err: Error) => {
      this.status = '连接失败：' + err.message;
    }).finally(() => {
      req.destroy();
      this.requesting = false;
    });
  }

  // Top-down display: forward x points up; lateral y points to the right.
  // Radar at the bottom-center; the circle and grid do not imply an angle sensor.
  private drawRadarMap(): void {
    if (!this.canvasReady) { return; }
    const ctx = this.context;
    const W: number = 320;
    const H: number = 232;
    const originX: number = 160;
    const originY: number = 211;
    const maxRangeM: number = 6000;
    const scale: number = 185 / maxRangeM;

    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = '#102A46';
    ctx.fillRect(0, 0, W, H);
    ctx.lineWidth = 1;
    ctx.strokeStyle = '#3D657F';
    // Forward-axis and lateral-axis grid.
    ctx.beginPath();
    ctx.moveTo(originX, 13);
    ctx.lineTo(originX, originY);
    ctx.moveTo(5, originY);
    ctx.lineTo(W - 5, originY);
    ctx.stroke();

    // The 2km, 4km and 6km circular arcs are range reference rings.
    for (let k = 1; k <= 3; k++) {
      const radius: number = k * 2000 * scale;
      ctx.beginPath();
      ctx.arc(originX, originY, radius, Math.PI, 2 * Math.PI);
      ctx.stroke();
    }

    ctx.fillStyle = '#AAC7DA';
    ctx.font = '11px sans-serif';
    ctx.fillText('前方 x', 165, 13);
    ctx.fillText('左侧 -y', 6, 229);
    ctx.fillText('右侧 +y', 263, 229);
    ctx.fillText('2km', 168, originY - 2000 * scale);
    ctx.fillText('4km', 168, originY - 4000 * scale);
    ctx.fillText('6km', 168, originY - 6000 * scale);

    ctx.fillStyle = '#F9C74F';
    ctx.beginPath();
    ctx.arc(originX, originY, 5, 0, 2 * Math.PI);
    ctx.fill();

    for (let i = 0; i < this.targets.length; i++) {
      const t: RadarTarget = this.targets[i];
      const px: number = originX + t.y_m * scale;
      const py: number = originY - t.x_m * scale;
      if (!Number.isFinite(px) || !Number.isFinite(py) ||
        px < 5 || px > W - 5 || py < 5 || py > originY) {
        continue;
      }
      ctx.fillStyle = '#34E5C3';
      ctx.beginPath();
      ctx.arc(px, py, 5, 0, 2 * Math.PI);
      ctx.fill();
      ctx.fillStyle = '#FFFFFF';
      ctx.font = '11px sans-serif';
      ctx.fillText((i + 1).toString(), px + 8, py - 5);
    }
  }

  build() {
    Scroll() {
      Column({ space: 14 }) {
        Text('雷达三参数监测看板')
          .fontSize(24)
          .fontWeight(FontWeight.Bold)
          .fontColor('#123044')
          .width('100%')

        Text('8接收天线 · 二维CA-CFAR · DBF方位角估计')
          .fontSize(12)
          .fontColor('#64748B')
          .width('100%')

        Text('通信状态：' + this.status)
          .fontSize(13)
          .fontColor('#087F72')
          .width('100%')
          .padding(10)
          .backgroundColor('#E3F5F1')
          .borderRadius(8)

        Row({ space: 10 }) {
          Column({ space: 4 }) {
            Text('当前帧').fontSize(12).fontColor('#64748B')
            Text(this.frameId.toString()).fontSize(23).fontWeight(FontWeight.Bold)
          }
          .layoutWeight(1)
          .padding(12)
          .backgroundColor(Color.White)
          .borderRadius(12)

          Column({ space: 4 }) {
            Text('检测目标数').fontSize(12).fontColor('#64748B')
            Text(this.targets.length.toString()).fontSize(23).fontWeight(FontWeight.Bold)
          }
          .layoutWeight(1)
          .padding(12)
          .backgroundColor(Color.White)
          .borderRadius(12)
        }

        Text('天气：' + this.weatherZH(this.weather))
          .fontSize(15)
          .width('100%')

        Text('目标二维位置图（x 前方，y 横向）')
          .fontSize(16)
          .fontWeight(FontWeight.Bold)
          .width('100%')

        Canvas(this.context)
          .width(320)
          .height(232)
          .backgroundColor('#102A46')
          .onReady(() => {
            this.canvasReady = true;
            this.drawRadarMap();
          })

        Text('目标检测列表')
          .fontSize(17)
          .fontWeight(FontWeight.Bold)
          .width('100%')

        if (this.targets.length === 0) {
          Text('当前没有可展示的三参数检测结果')
            .fontSize(13)
            .fontColor('#64748B')
            .width('100%')
        }

        ForEach(this.targets, (t: RadarTarget, i: number) => {
          Column({ space: 6 }) {
            Text('目标 ' + (i + 1))
              .fontSize(16)
              .fontWeight(FontWeight.Bold)
              .fontColor('#087F72')
            Text('距离：' + t.range_m.toFixed(1) + ' m').fontSize(14)
            Text('径向速度：' + t.velocity_mps.toFixed(2) + ' m/s').fontSize(14)
            Text('方位角：' + t.azimuth_deg.toFixed(1) + '°').fontSize(14)
            Text('前向 x：' + t.x_m.toFixed(1) + ' m').fontSize(14)
            Text('横向 y：' + t.y_m.toFixed(1) + ' m').fontSize(14)
            Text('相对峰值：' + t.peak_db.toFixed(1) + ' dB').fontSize(14)
          }
          .width('100%')
          .alignItems(HorizontalAlign.Start)
          .padding(14)
          .backgroundColor(Color.White)
          .borderRadius(10)
        }, (t: RadarTarget, i: number) => i.toString())

        Button('立即刷新')
          .height(44)
          .width('100%')
          .backgroundColor('#0D9488')
          .onClick(() => { this.fetchLatest(); })
      }
      .width('100%')
      .padding(16)
      .alignItems(HorizontalAlign.Center)
    }
    .width('100%')
    .height('100%')
    .backgroundColor('#F1F5F9')
  }
}
