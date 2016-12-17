<template>
  <!-- Main content -->
  <section class='content'>
    <!-- Info boxes -->
     <!--
    YYY
    -->
    <div class='col-xs-12'>
      <div class="box">
        <div class="box-header with-border">
          <h3 class="box-title"></h3>
          <div class="box-body">
              <div class="col-sm-6 col-xs-12">
              <p class="text-center">
                <strong>Summary</strong>
              </p>
          <!-- /.box-header -->
            <table class="table table-striped">
              <tbody>
                <tr v-for="item in summary">
                    <th>{{ item.key }}</th>
                    <th>{{ item.value }}</th>
                </tr>
              </tbody>
            </table>
        </div>
          <!-- /.box-body -->
            <!--
            <hr class="visible-xs-block">
            <div class="col-sm-6 col-xs-12">
              <p class="text-center">
                <strong>Language Overview</strong>
              </p>
              <canvas id="languagePie"></canvas>
            </div>
-->
          </div>
        </div>
        <!--
        <small class="space"><b>Pro Tip</b> Don't forget to star us on github!</small>
        -->
      </div>
    </div>
    <!-- /.row -->
    <div class='row'>
            <div class="col-sm-6 col-xs-12" v-show="showGroupBars">
              <p class="text-center">
                <strong>Count by Group</strong>
              </p>
              <canvas id="groupBars" ></canvas>
            </div>
            <div class="col-sm-6 col-xs-12" v-show="showLabel1Bars">
              <p class="text-center">
                <strong>Count by Label</strong>
              </p>
              <canvas id="label1Bars" ></canvas>
            </div>
            <div class="col-sm-6 col-xs-12" v-show="showLabel2Bars">
              <p class="text-center">
                <strong>Count by Label2</strong>
              </p>
              <canvas id="label2Bars" ></canvas>
            </div>
            <div class="col-sm-6 col-xs-12" v-show="showShapeBars">
              <p class="text-center">
                <strong>Count by Shape</strong>
              </p>
              <canvas id="shapeBars" ></canvas>
            </div>
    </div>

    <!-- Main row -->
    <!--
    <div class='row'>
      <div class='col-md-3 col-sm-6 col-xs-12'>
        <div class='info-box bg-yellow'>
          <span class='info-box-icon'><i class='ion ion-ios-pricetag-outline'></i></span>

          <div class='info-box-content'>
            <span class='info-box-text'>Inventory</span>
            <span class='info-box-number'>5,200</span>

            <div class='progress'>
              <div class='progress-bar' style='width: 50%'></div>
            </div>
                <span class='progress-description'>
                  50% Increase
                </span>
          </div>
        </div>
      </div>
      <div class='col-md-3 col-sm-6 col-xs-12'>
        <div class='info-box bg-green'>
          <span class='info-box-icon'><i class='ion ion-ios-heart-outline'></i></span>

          <div class='info-box-content'>
            <span class='info-box-text'>Mentions</span>
            <span class='info-box-number'>92,050</span>

            <div class='progress'>
              <div class='progress-bar' style='width: 20%'></div>
            </div>
                <span class='progress-description'>
                  20% Increase
                </span>
          </div>
        </div>
      </div>
      <div class='col-md-3 col-sm-6 col-xs-12'>
        <div class='info-box bg-red'>
          <span class='info-box-icon'><i class='ion ion-ios-cloud-download-outline'></i></span>

          <div class='info-box-content'>
            <span class='info-box-text'>Downloads</span>
            <span class='info-box-number'>114,381</span>

            <div class='progress'>
              <div class='progress-bar' style='width: 70%'></div>
            </div>
                <span class='progress-description'>
                  70% Increase
                </span>
          </div>
        </div>
      </div>
      <div class='col-md-3 col-sm-6 col-xs-12'>
        <div class='info-box bg-aqua'>
          <span class='info-box-icon'><i class='ion-ios-chatbubble-outline'></i></span>

          <div class='info-box-content'>
            <span class='info-box-text'>Direct Messages</span>
            <span class='info-box-number'>163,921</span>

            <div class='progress'>
              <div class='progress-bar' style='width: 40%'></div>
            </div>
                <span class='progress-description'>
                  40% Increase
                </span>
          </div>
        </div>
      </div>
    </div>
    -->
    <!-- /.row -->
  </section>
  <!-- /.content -->
</template>

<script>
import $ from 'jquery'
import Chart from 'chart.js'

function showBars (el, data) {
  console.log(el)
  var labels = []
  var bars = []
  for (var i in data) {
    var l = data[i]
    labels.push(l[0])
    bars.push(l[1])
  }
  console.log(labels)
  console.log(bars)
  console.log(document.getElementById(el))
  var ctx = document.getElementById(el).getContext('2d')
  var config = {
    type: 'bar',
    data: {
      labels: labels,
      datasets: [{
        borderColor: '#284184',
        // pointBackgroundColor: '#284184',
        backgroundColor: 'rgba(0, 0, 0, 0)',
        borderWidth: 1,
        data: bars
      }]
    },
    options: {
      responsive: true,
      scales: {
        yAxes: [{
          ticks: {
            min: 0,
            beginAtZero: true
          }
        }]
      },
      // maintainAspectRatio: !this.isMobile,
      legend: {
        position: 'bottom',
        display: false
      }
      /*
      tooltips: {
        // mode: 'label',
        xPadding: 10,
        yPadding: 10,
        bodySpacing: 10
      }
      */
    }
  }

  new Chart(ctx, config) // eslint-disable-line no-new
}

module.exports = {
  data: function () {
    return {
      summary: [],
      showGroupBars: false,
      showLabel1Bars: false,
      showLabel2Bars: false,
      showShapeBars: false
    }
  },
  mounted: function () {
    var thisData = this
    this.$nextTick(function () {
      $.getJSON('/api/overview', function (overview) {
        console.log(overview)
        thisData.summary = overview.summary
        if ('group_cnt' in overview) {
          var groupBars = overview['group_cnt']
          thisData.showGroupBars = true
          showBars('groupBars', groupBars)
        }
        if ('label1_cnt' in overview) {
          var label1Bars = overview['label1_cnt']
          thisData.showLabel1Bars = true
          showBars('label1Bars', label1Bars)
        }
        if ('label2_cnt' in overview) {
          var label2Bars = overview['label2_cnt']
          thisData.showLabel2Bars = true
          showBars('label2Bars', label2Bars)
        }
        if ('shape_cnt' in overview) {
          var shapeBars = overview['shape_cnt']
          thisData.showShapeBars = true
          showBars('shapeBars', shapeBars)
        }
      })
    })
  }
}
</script>
<style>
.info-box {
  cursor: pointer;
}
.info-box-content {
  text-align: center;
  vertical-align: middle;
  display: inherit;
}
.fullCanvas {
  width: 100%;
}
</style>
