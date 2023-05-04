<template>
  <div class="camera_root">
    <el-card :style="{height:screenHeight+'px'}">
      <baidu-map
        id="allmap"
        :style="{display:'flex','flex-direction': 'column-reverse',width: '100%',height:(screenHeight-50)+'px'}"
        :scroll-wheel-zoom="true"
        @ready="mapReady"
        @click="getLocation"
      >
        <el-divider />
        <div style="display:flex;justify-content:center;margin-bottom: 5px">
          <bm-auto-complete v-model="searchJingwei" :sug-style="{zIndex: 999999}">
            <el-input v-model="searchJingwei" style="width:300px;margin-right:15px" placeholder="输入地址" />
          </bm-auto-complete>
          <el-button type="primary" @click="getBaiduMapPoint">搜索</el-button>
        </div>
        <bm-map-type :map-types="['BMAP_NORMAL_MAP', 'BMAP_HYBRID_MAP']" anchor="BMAP_ANCHOR_TOP_LEFT" />
        <bm-marker v-if="infoWindowShow" :position="{lng: longitude, lat: latitude}">
          <bm-label content="" :label-style="{color: 'red', fontSize : '24px'}" :offset="{width: -35, height: 30}" />
        </bm-marker>
        <bm-info-window style="height: 200px;overflow: auto" :position="{lng: longitude, lat: latitude}" :show="infoWindowShow" @clickclose="infoWindowClose">
          <img class="srcimg" :src="getBaseUrl() + value.url" title="点击预览" @click="handlePictureCardPreview(value.url)">
          <img class="modelimg" :src="getBaseUrl() + value.modelurl" title="点击预览" @click="handlePictureCardPreview(value.modelurl)">
          <div>纬度:{{ this.latitude }}</div>
          <div>经度:{{ this.longitude }}</div>
        </bm-info-window>
      </baidu-map>
    </el-card>
    <!--  单条道路质量评估报告  -->
    <el-card :style="{height:screenHeight+'px',display: 'flex','flex-direction':'column',overflow:'auto'}">
      <div slot="header" class="clearfix">
        <span>道路检测报告</span>
        <el-button style="float: right; padding: 3px 0" type="text" @click="getPdf('页面导出PDF文件名')">报告导出</el-button>
      </div>
      <!--   表格   -->
      <div class="body" id="pdfDom">
        <!--        <el-divider>数据集</el-divider>-->
        <el-table
          :data="mapDataList"
          style="width: 100%">
          <el-table-column
            prop="user_id"
            label="用户id">
          </el-table-column>
          <el-table-column
            prop="check_type"
            label="检测类型">
            <template slot-scope="scope">
              {{scope.row.check_type=='img'?"图片":"视频"}}
            </template>
          </el-table-column>
          <el-table-column
            prop="check_time"
            label="检测时间">
            <template slot-scope="scope">
              {{handlerTime(scope.row.check_time)}}
            </template>
          </el-table-column>
          <el-table-column label="原始图片/视频" align="center">
            <template slot-scope="scope">
              <img v-if="scope.row.check_type=='img'" style="width: 150px;height: auto;cursor: pointer" :src="getBaseUrl()+scope.row.srcurl" title="点击预览" @click="handlePictureCardPreview(scope.row.srcurl,'img')">
              <video v-else style="width: 150px;height: auto" :src="getBaseUrl()+scope.row.srcurl" crossOrigin="Anonymous" @click="handlePictureCardPreview(scope.row.srcurl,'video')" />
            </template>
          </el-table-column>
          <el-table-column label="检测结果" align="center">
            <template slot-scope="scope">
              <img v-if="scope.row.check_type=='img'" style="width: 150px;height: auto;cursor: pointer" :src="getBaseUrl()+scope.row.url" title="点击预览" @click="handlePictureCardPreview(scope.row.url,'img')">
              <video v-else style="width: 150px;height: auto" :src="getBaseUrl()+scope.row.url" crossOrigin="Anonymous" @click="handlePictureCardPreview(scope.row.url,'video')" />
            </template>
          </el-table-column>
        </el-table>
      </div>
    </el-card>

    <el-dialog :visible.sync="dialogVisible">
      <img width="100%" :src="dialogImageUrl" alt="" title="点击在新窗口预览" @click="openImg">
    </el-dialog>
  </div>
</template>

<script>
import * as echarts from 'echarts'
import deviceMixin from '@/mixins/device'
import apiBaseUrl from '@/mixins/apiBaseUrl'
import { getList } from '@/api/checklist'
export default {
  mixins: [deviceMixin, apiBaseUrl],
  data() {
    return {
      searchJingwei: '',
      infoWindowShow: false,
      longitude: '',
      latitude: '',
      point: '',
      dialogVisible: false,
      dialogImageUrl: '',
      value: {
        url: '/static/img/testres.jpg',
        modelurl: '/static/resImg/testres.jpg'
      },
      mapDataList: [],
      chart: null
    }
  },
  mounted() {
    this.initChart()
  },
  methods: {
    handlerTime(time) {
      return String(new Date(parseInt(time))).slice(10,-10)
    },
    fetchData(name) {
      this.listLoading = true
      getList().then(response => {
        this.mapDataList = response.data.items.slice(0, 3)
      }).catch(() => {
        console.log('获取数据错误')
      })
    },
    // 地图初始化
    mapReady({ BMap, map }) {
      // 选择一个经纬度作为中心点
      this.point = new BMap.Point(113.27, 23.13)
      map.centerAndZoom(this.point, 12)
      this.BMap = BMap
      this.map = map
    },
    // 点击获取经纬度
    getLocation(e) {
      this.longitude = e.point.lng
      this.latitude = e.point.lat
      this.infoWindowShow = true
      this.fetchData()
    },
    getBaiduMapPoint() {
      const that = this
      const myGeo = new this.BMap.Geocoder()
      this.fetchData()
      myGeo.getPoint(this.searchJingwei, function(point) {
        if (point) {
          that.map.centerAndZoom(point, 15)
          that.latitude = point.lat
          that.longitude = point.lng
          // that.infoWindowShow = true
        }
      })
    },
    // 信息窗口关闭
    infoWindowClose() {
      this.latitude = ''
      this.longitude = ''
      this.infoWindowShow = false
    },

    handlePictureCardPreview(url) {
      url = this.getBaseUrl() + url
      this.dialogImageUrl = url
      this.dialogVisible = true
    },
    openImg() {
      window.open(this.dialogImageUrl)
    },
    initChart() {
      this.chart = echarts.init(document.getElementById('base-echart'))
      this.chart.setOption({
        xAxis: {
          type: 'category',
          data: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        },
        yAxis: {
          type: 'value'
        },
        series: [
          {
            data: [150, 230, 224, 218, 135, 147, 260],
            type: 'line'
          }
        ]
      })
      window.addEventListener('resize', () => {
        this.chart.resize()
        this.chart2.resize()
      })
    }
  }
}
</script>

<style scoped lang="scss">
.camera_root{
  height: 500px;
  display: flex;

  .el-card:nth-of-type(1){
    flex: 4;
    margin: 0px 10px;
  }
  .el-card:nth-of-type(2){
    flex: 3;
  }

  .imgs {
    width: 100%;
    height: 100%;
    display: flex;
    img {
      width: 50%;
    }
    img:nth-child(1){
      margin-right: 10px;
    }
  }

  .bm-view{
    width: 100%;
    height: 100%;
  }
  #allmap{
    height: 450px;
    width: 100%;
    margin: 10px 0;
  }
  .srcimg ,.modelimg{
    width: 100%;
    height: 80%;
  }
}
</style>
