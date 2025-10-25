import { Camera, Database, Sparkles, ScanSearch, Eye, ArrowRight } from 'lucide-react'

const Flow = () => {
  const steps = [
    { icon: <Camera className="w-7 h-7" />, title: 'Captura', color: 'from-cyan-500 to-blue-500' },
    { icon: <Database className="w-7 h-7" />, title: 'Entrenamiento', color: 'from-green-500 to-emerald-500' },
    { icon: <Sparkles className="w-7 h-7" />, title: 'Extracción', color: 'from-blue-500 to-indigo-500' },
    { icon: <ScanSearch className="w-7 h-7" />, title: 'Detección', color: 'from-orange-500 to-red-500' },
    { icon: <Eye className="w-7 h-7" />, title: 'Visualización', color: 'from-indigo-500 to-cyan-500' }
  ]

  return (
    <div className="py-12">
      <div className="text-center mb-8">
        <div className="inline-flex items-center gap-2 bg-cyan-500/10 border border-cyan-500/30 rounded-full px-5 py-2 mb-4">
          <Sparkles className="w-4 h-4 text-cyan-400" />
          <span className="text-cyan-400 text-sm font-semibold tracking-wide">PROCESO</span>
        </div>
        <h3 className="text-3xl md:text-4xl font-black text-white mb-3">
          Flujo de Detección
        </h3>
        <p className="text-slate-400">Pipeline completo de procesamiento</p>
      </div>

      {/* Desktop View */}
      <div className="hidden md:flex flex-wrap justify-center items-center gap-4">
        {steps.map((step, index) => (
          <div key={index} className="flex items-center gap-4">
            <div className="group relative">
              <div className={`absolute inset-0 bg-gradient-to-br ${step.color} rounded-full blur-xl opacity-75 group-hover:opacity-100 transition duration-300`}></div>
              <div className="relative flex flex-col items-center">
                <div className={`bg-gradient-to-br ${step.color} text-white p-4 rounded-full shadow-lg mb-2`}>
                  {step.icon}
                </div>
                <p className="font-semibold text-white text-sm">{step.title}</p>
              </div>
            </div>
            {index < steps.length - 1 && (
              <ArrowRight className="w-6 h-6 text-slate-600" />
            )}
          </div>
        ))}
      </div>

      {/* Mobile View */}
      <div className="md:hidden space-y-4">
        {steps.map((step, index) => (
          <div key={index}>
            <div className="group relative">
              <div className={`absolute inset-0 bg-gradient-to-br ${step.color} rounded-2xl blur-xl opacity-0 group-hover:opacity-50 transition duration-500`}></div>
              <div className="relative bg-gradient-to-br from-slate-800 to-slate-900 border border-slate-700 rounded-2xl p-4 hover:border-cyan-500/50 transition-all duration-300 flex items-center gap-4">
                <div className={`bg-gradient-to-br ${step.color} text-white p-3 rounded-xl`}>
                  {step.icon}
                </div>
                <div>
                  <span className="text-xs text-slate-500 font-semibold">Paso {index + 1}</span>
                  <p className="font-bold text-white">{step.title}</p>
                </div>
              </div>
            </div>
            {index < steps.length - 1 && (
              <div className="flex justify-center my-2">
                <div className="text-2xl text-slate-600">↓</div>
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  )
}

export default Flow